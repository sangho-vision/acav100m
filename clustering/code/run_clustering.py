import copy
from collections import defaultdict
from pathlib import Path
import os
import math

import torch
import numpy as np
import webdataset as wds
from braceexpand import braceexpand
from tqdm import tqdm

from models import get_model, get_model_class
from utils import (
    dump_pickle, to_device, dol_to_lod, get_data_cache_path,
    filter_models
)
import mps.distributed as du
from sgd_clustering import KMeans
from data.clustering import get_clustering_dataloader
from save import _save_output, store_shards_set
from process_batch import extract_batch_cached, train_batch_cached


def run_clustering(args):
    model_names = filter_models(args)
    clusterings = train_clusters(args, model_names)
    save_paths = assign_clusters(args, model_names, clusterings)
    return save_paths


def init_clusterings(args, model_classes):
    clusterings = {}
    for model_name, model_class in model_classes.items():
        ds = model_class.output_dims
        if isinstance(ds, int):
            clusterings[model_name] = {'model': KMeans(args, ds, args.clustering.ncentroids)}
        else:
            # layer_model
            clusterings[model_name] = {f'layer_{i}': KMeans(args, d, args.clustering.ncentroids)
                                       for i, d in enumerate(ds)}

    clusterings = _init_clusterings(args, clusterings)
    return clusterings


def _init_clusterings(args, clusterings):
    for model_name in clusterings.keys():
        for key in clusterings[model_name].keys():
            clusterings[model_name][key].to(args.computation.device)
            clusterings[model_name][key].initialize() # For DistributedDataParallel
    return clusterings


def load_clusterings(args, model_classes):
    not_loaded = True
    if hasattr(args.clustering, 'cached_epoch') and isinstance(args.clustering.cached_epoch, int):
        epoch = copy.deepcopy(args.clustering.cached_epoch)
        path, name = get_data_cache_path(args)
        out_path = path / "cache_epoch_{}_{}".format(epoch, name)
        if out_path.is_file():
            clusterings, not_loaded = load_path(args, out_path, model_classes)
        elif args.clustering.load_cache_from_shard_subset:
            out_path = get_shard_subset_cache(args, path, epoch, name)
            if out_path is None:
                not_loaded = True
            else:
                clusterings, not_loaded = load_path(args, out_path, model_classes)
        if not_loaded:
            print("no clustering cache found.")
    if not_loaded:
        clusterings = init_clusterings(args, model_classes)
    return clusterings, not not_loaded


def get_shard_subset_cache(args, path, epoch, name):
    all_caches = list(path.glob("cache_epoch_{}_*.pkl".format(epoch)))
    all_caches = {p: set(braceexpand(p.name[p.name.find('shard-'):])) for p in all_caches}
    shard_set = set(braceexpand(name))
    all_caches = {p: v for p, v in all_caches.items() if len(v - shard_set) == 0}
    if len(all_caches) == 0:
        return None
    path = max(list(all_caches.items()), key=lambda v: v[1])[0]
    return path


def load_path(args, out_path, model_classes):
    not_loaded = True
    if out_path.is_file():
        clusterings = torch.load(str(out_path))
        clusterings = _load_clusterings(args, clusterings)
        if len(set(model_classes.keys()) - set(clusterings.keys())) == 0:
            print("loading from clustering cache: {}".format(out_path))
            clusterings = _init_clusterings(args, clusterings)
            not_loaded = False
        else:
            print("clustering cache features does not match with the given models")
    else:
        print("no clustering cache file found: {}".format(out_path))
    return clusterings, not_loaded


def _load_clusterings(args, clusterings):
    if args.clustering.save_scheme_ver2:
        clusterings = {k: {k2: KMeans.load(v2) for k2, v2 in v.items()} for k, v in clusterings.items()}
        clusterings = _init_clusterings(args, clusterings)
    return clusterings


def save_clusterings(args, epoch, clusterings):
    path, name = get_data_cache_path(args)
    out_path = path / "cache_epoch_{}_{}".format(epoch, name)
    tqdm.write("saving clustering cache to: {}".format(out_path))
    if args.clustering.save_scheme_ver2:
        clusterings = {k: {k2: v2.get_attrs() for k2, v2 in v.items()} for k, v in clusterings.items()}
    torch.save(clusterings, str(out_path))


def get_model_data(args, model_names):
    model_classes = {}
    model_args = {}
    model_key_map = {}
    for model_name in model_names:
        model_class, model_arg = get_model_class(model_name, args)
        model_classes[model_name] = model_class
        model_args[model_name] = model_arg
        tag = model_class.model_tag
        model_key_map[model_name] = '/'.join([tag['name'], tag['dataset']])
    return model_classes, model_args, model_key_map


def train_clusters(args, model_names):
    model_classes, model_args, model_key_map = get_model_data(args, model_names)

    clusterings, loaded = load_clusterings(args, model_classes)
    if loaded and not args.clustering.resume_training:
        return clusterings

    dataloader = get_clustering_dataloader(args, drop_last=True, shuffle=True, is_train=True)
    tqdm.write("training sgd kmeans for models: {}".format(model_names))

    pre_epochs = 0
    if loaded:
        pre_epochs = copy.deepcopy(args.clustering.cached_epoch)

    clustering_epochs = math.ceil(args.clustering.epochs / args.computation.num_gpus)

    ids = []
    shard_sizes = {}
    pivot_name = model_names[0]
    shards = {model_name: defaultdict(dict) for model_name in model_names}
    ids = defaultdict(list)
    saved_paths = []
    with torch.no_grad():
        if isinstance(dataloader, (wds.MultiDataset, )):
            iters = len(dataloader)
        else:
            iters = len(dataloader) \
                if hasattr(dataloader.dataset, '__len__') \
                and dataloader.dataset.__len__() is not None else None
        if iters is not None:
            tqdm.write('begin with {} iters'.format(iters))
        cnt = 0
        for epoch in tqdm(range(pre_epochs, clustering_epochs + pre_epochs), desc='kmeans_epoch'):
            # set lr
            for model_name, clustering in clusterings.items():
                for k in clustering.keys():
                    clusterings[model_name][k].lr = 0.1 ** (2 + epoch // 5)

            for batch in tqdm(dataloader):
                distances = {}
                for model_name in model_names:
                    model_batch = batch[model_key_map[model_name]]
                    distance = train_batch_cached(args, model_batch, clusterings[model_name])
                    distances[model_name] = distance
            save_clusterings(args, epoch, clusterings)
    return clusterings


def assign_clusters(args, model_names, clusterings):
    model_classes, model_args, model_key_map = get_model_data(args, model_names)
    dataloader = get_clustering_dataloader(args, drop_last=False, shuffle=False, is_train=False)

    tqdm.write("extracting clustering for models: {}".format(model_names))

    ids = []
    shard_sizes = {}
    pivot_name = model_names[0]
    shards = {model_name: defaultdict(dict) for model_name in model_names}
    ids = defaultdict(list)
    saved_paths = []
    num_saved = 0

    def save_shard(shard_name):
        nonlocal num_saved

        data = []
        for model_name, model_class in model_classes.items():
            datum = {'model_key': model_name, 'data': shards[model_name][shard_name],
                        **model_class.model_tag}
            data.append(datum)
        prefix = ''
        if args.clustering.cached_epoch is not None:
            prefix = 'epoch_{}_'.format(args.clustering.cached_epoch)
        saved_path = _save_output(args, shard_name, ids[shard_name], data,
                                    name='assignments',
                                    prefix=prefix)
        saved_paths.append(saved_path)
        del ids[shard_name]
        del shard_sizes[shard_name]
        for model_name in shards.keys():
            del shards[model_name][shard_name]
        num_saved += 1

    with torch.no_grad():
        if isinstance(dataloader, (wds.MultiDataset, )):
            iters = len(dataloader)
        else:
            iters = len(dataloader) \
                if hasattr(dataloader.dataset, '__len__') \
                and dataloader.dataset.__len__() is not None else None
        if iters is not None:
            tqdm.write('begin with {} iters'.format(iters))
        cnt = 0
        for batch in dataloader:
            model_features = {}
            for model_name in model_names:
                model_batch = batch[model_key_map[model_name]]
                features = extract_batch_cached(args, batch, model_batch, clusterings[model_name])
                model_features[model_name] = features

            features = dol_to_lod(model_features)
            for feature in features:
                idx = feature[pivot_name].get('idx')
                shard_name = feature[pivot_name].get('shard_name')
                if idx not in ids[shard_name]:
                    ids[shard_name].append(idx)
                    shard_size = feature[pivot_name].get('shard_size')
                    shard_sizes[shard_name] = shard_size
                    for model_name in model_names:
                        shards[model_name][shard_name][idx] = feature[model_name]

            del model_features
            del batch

            shard_names = list(ids.keys())
            for shard_name in shard_names:
                out_path = args.data.output.path / (shard_name + '.pkl')
                if Path(out_path).is_file():
                    continue

                if shard_name in shard_sizes:
                    if len(ids[shard_name]) >= (shard_sizes[shard_name]):
                        save_shard(shard_name)

            if args.debug:
                break
            cnt += 1

    tqdm.write(f"node{du.get_rank()}: {cnt} iters")
    shard_names = list(ids.keys())
    num_not_full = 0
    for shard_name in shard_names:
        if shard_name in shard_sizes:
            if len(ids[shard_name]) >= round(shard_sizes[shard_name] * args.data.output.shard_ok_ratio):
                if len(ids[shard_name]) < shard_sizes[shard_name]:
                    num_not_full += 1
                save_shard(shard_name)

    tqdm.write(f"node{du.get_rank()}: (num_saved_shards: {num_saved}), (num_full_shards: {num_saved - num_not_full})")

    return saved_paths
