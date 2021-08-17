from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import os
import time

import numpy as np

import torch
import webdataset as wds

from models import get_model
from data.loader import get_dataloader, collate
from process_batch import extract_batch
from utils import (
    dump_pickle, dol_to_lod, get_idx,
    filter_models
)
from save import save_output, load_cache_features, save_cache
import mps.distributed as du


def run_extraction(args):
    model_names = filter_models(args)
    save_paths = _run_extraction(args, model_names)
    return save_paths


def _run_extraction(args, model_names):
    models = {}
    model_args = {}
    for model_name in model_names:
        model, model_arg = get_model(model_name, args)
        models[model_name] = model
        model_args[model_name] = model_arg
    dataloader = get_dataloader(args, model=models)

    tqdm.write("extracting feature for models: {}".format(model_names))

    shard_sizes = {}
    pivot_name = model_names[0]
    shards = {model_name: defaultdict(dict) for model_name in model_names}
    ids = defaultdict(list)
    saved_paths = []
    num_saved = 0

    if hasattr(dataloader.dataset, 'caches') and dataloader.dataset.caches is not None:
        caches = dataloader.dataset.caches
        for shard_name, cache in caches.items():
            ids[shard_name] = [get_idx(row['filename']) for row in cache]
            shard_size = {row['shard_name']: row['shard_size'] for row in cache}
            shard_sizes = {**shard_sizes, **shard_size}
            for model_name in model_names:
                shards[model_name][shard_name] = load_cache_features(caches, model_name, shard_name)

    def save_shard(shard_name):
        nonlocal num_saved
        saved_path = save_output(args, models, shards, ids, shard_name, final=True)
        saved_paths.append(saved_path)
        del ids[shard_name]
        del shard_sizes[shard_name]
        for model_name in shards.keys():
            del shards[model_name][shard_name]
        num_saved += 1

    # from pympler import tracker; tr = tracker.SummaryTracker()
    with torch.no_grad():
        if isinstance(dataloader, (wds.MultiDataset, )):
            iters = len(dataloader)
        else:
            iters = len(dataloader) if dataloader.dataset.__len__() is not None else None
        if iters is not None:
            tqdm.write('begin with {} iters'.format(iters))
        cnt = 0
        batch_time_lst = []
        end_time = time.time()
        for n_iter, batch in enumerate(dataloader):
            model_features = {}
            for model_name in model_names:
                model_batch = [row[model_name] for row in batch]
                model_batch, options = collate(model_batch)
                features = extract_batch(model_args[model_name],
                                        models[model_name],
                                        model_batch, options)
                model_features[model_name] = features

            features = dol_to_lod(model_features)
            for feature in features:
                shard_name = feature[pivot_name].get('shard_name')
                idx = feature[pivot_name].get('idx')
                if idx not in ids[shard_name]:
                    ids[shard_name].append(idx)
                    shard_size = feature[pivot_name].get('shard_size')
                    shard_sizes[shard_name] = shard_size
                    for model_name in model_names:
                        shards[model_name][shard_name][idx] = feature[model_name]

            del model_features
            del batch

            # save data
            shard_names = list(ids.keys())
            for shard_name in shard_names:
                out_path = args.data.output.path / (shard_name + '.pkl')
                if (n_iter + 1) % args.acav.save_cache_every == 0:
                    save_cache(args, models, shards, ids, shard_name)
                if Path(out_path).is_file():
                    continue

                if shard_name in shard_sizes:
                    if len(ids[shard_name]) >= (shard_sizes[shard_name]):
                        save_shard(shard_name)

            cnt += 1
            batch_time = time.time() - end_time
            batch_time_lst.append(batch_time)
            if args.debug or (du.is_master_proc() and cnt % args.log_period == 0):
                tqdm.write(f"node{du.get_rank()}: {cnt}/{iters} ({int(batch_time)} secs elapsed, mean: {int(np.mean(batch_time_lst))} secs) iters processed...")
            end_time = time.time()

    # finished
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
