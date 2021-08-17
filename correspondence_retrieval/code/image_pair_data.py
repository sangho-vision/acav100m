import copy
import math
import random
from itertools import chain
from collections import defaultdict, OrderedDict
from pathlib import Path

from tqdm import tqdm

import numpy as np
import torch

from utils import load_pickle, dump_pickle, merge_dataset_model_name, peek
from model import get_model
from image_datasets import dataset_dict
from derangement import derangement


def get_loader(dataset, num_workers):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=num_workers)
    loader.name = dataset.name
    return loader


def get_features(args, dataset_names, device, num_workers, finetune=False, sample=False,
                 model_name='ResNet52', image_pair_data_path='./data/image_pair_data',
                 chunk_size=1000, extract_each_layer=False):
    dataset_names = [name for name in dataset_names if name in args.data_requires_extraction]
    for dataset_name in sorted(dataset_names):
        loader = get_loader(dataset_dict[dataset_name](image_pair_data_path), num_workers=num_workers)
        get_feature(loader, device, num_workers=num_workers,
                    model_name=model_name,
                    image_pair_data_path=image_pair_data_path,
                    dataset_name=dataset_name,
                    chunk_size=chunk_size,
                    extract_each_layer=extract_each_layer)


def get_feature(loader, device, num_workers, finetune=False, sample=False,
                model_name='ResNet52', image_pair_data_path='./data/image_pair_data',
                dataset_name='mnist', chunk_size=1000, extract_each_layer=False):
    models = get_model(device, num_workers, finetune=finetune, sample=sample,
                       model_name=model_name, extract_each_layer=extract_each_layer)
    for extractor_name, model in models.items():
        full_model_name = f'{model_name}_{extractor_name}'
        output_dir = Path(image_pair_data_path) / dataset_name / full_model_name
        output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
        if not chunks_full(output_dir, loader, chunk_size=chunk_size):
            tqdm.write(f"extracting feature for model: {full_model_name}, data: {dataset_name}")
            _get_feature(output_dir, model, loader, device, chunk_size=chunk_size)


def chunks_full(output_dir, loader, chunk_size=1000):
    num_chunks = len(list(output_dir.glob('chunk_*.pkl')))
    num_datapoints = len(loader.dataset)
    num_all_chunks = math.ceil(num_datapoints / chunk_size)
    return num_chunks == num_all_chunks


def _get_feature(output_dir, model, loader, device, chunk_size=1000):

    def save_chunk(to_save, chunk_num):
        chunk_path = output_dir / f'chunk_{chunk_num}.pkl'
        tqdm.write(f"saving chunk: {chunk_path}")
        dump_pickle(to_save, chunk_path)

    chunk_num = 0
    chunk = OrderedDict()
    chunk_size = chunk_size
    ids_pointer = 0
    with torch.no_grad():
        for data, labels in tqdm(loader, ncols=80):
            data = data.to(device)
            if data.shape[1] == 1:  # MNIST - grayscale
                data = data.repeat(1, 3, *[1 for _ in range(data.dim() - 2)])
            features = model(data)
            features = list(features.detach().cpu().numpy())
            features = [{'features': f, 'label': l.item()} for f, l in zip(features, labels)]
            num_points = len(labels)
            ids = list(range(ids_pointer, ids_pointer + num_points))
            ids_pointer += num_points
            chunk.update(dict(zip(ids, features)))
            del data
            if len(chunk) >= chunk_size:
                chunk = list(chunk.items())
                to_save, chunk = chunk[:chunk_size], chunk[chunk_size:]
                chunk = OrderedDict(chunk)
                to_save = dict(to_save)
                save_chunk(to_save, chunk_num)
                chunk_num += 1

    if len(chunk) > 0:
        chunk_num += 1
        chunk = dict(chunk)
        save_chunk(chunk, chunk_num)

    del model
    del loader


def remove_labels(views):
    labels = {view: [x['label'] for x in classes] for view, classes in views.items()}
    features = {view: [x['features'] for x in classes] for view, classes in views.items()}
    return features, labels


def remove_labels_view(view):
    return [x['features'] for x in view]


def cut(dt, length):
    keys = sorted(list(dt.keys()))
    return {key: dt[key] for key in keys[:length]}


def load_chunks(path):
    chunks = list(path.glob('chunk_*.pkl'))
    data = [load_pickle(path).items() for path in chunks]
    data = dict(chain(*data))  # merge chunks
    return data


def get_image_pair_data(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_names = get_dataset_names(args.data_name)
    get_features(args, dataset_names, device, args.num_workers, args.finetune, args.sample,
                 args.model_name, args.image_pair_data_path,
                 args.chunk_size, args.extract_each_layer)
    return _get_image_pair_data(args, dataset_names)


def get_dataset_names(data_name):
    name_dt = {
        'image_pair_mnist': ['cifar10', 'mnist'],
        'image_pair_rotation': ['cifar10', 'cifar10-rotated'],
        'image_pair_flip': ['cifar10', 'cifar10-flipped'],
        'image_pair_mnist_sound': ['mnist', 'fdss'],
        'kinetics_sounds': ['kinetics-sounds-slowfast', 'kinetics-sounds-vggish'],
    }
    data_name = data_name.lower()
    assert data_name in name_dt, f"invalid data pair type: {data_name}"
    return name_dt[data_name]


def _get_image_pair_data(args, dataset_names):
    path = Path(args.image_pair_data_path).resolve()
    paths = list(chain(*[(path / name).glob('*') for name in dataset_names]))
    paths = {merge_dataset_model_name(p.parent.stem, p.stem): p for p in paths}
    if args.extract_each_layer:
        paths = {k: p for k, p in paths.items() if p.stem.split('_')[-1] != 'model'}
    else:
        paths = {k: p for k, p in paths.items() if p.stem.split('_')[-1] == 'model'}
    data = {k: load_chunks(v) for k, v in paths.items()}
    # train split does not support sharding
    train, test = \
        derangement(data, deranged_classes_ratio=args.deranged_classes_ratio,
                    shuffle_true_ids=args.shuffle_true_ids,
                    class_datapoints_threshold=args.nsamples_per_class,
                    shuffle_datapoints=args.shuffle_datapoints,
                    shuffle_each_cluster=args.shuffle_each_cluster,
                    num_shards=args.num_shards,
                    train_ratio=args.train_ratio,
                    sample_level=args.sample_level)
    if train is not None:
        train = postprocess_train(train, args.sample_level)
    return train, postprocess(test)


def postprocess(shards):
    res = {'unsharded': _postprocess(*shards['unsharded'])}
    if 'sharded' in shards:
        res['sharded'] = [_postprocess(*shard) for shard in shards['sharded']]
    if 'sharded_ids' in shards:
        res['sharded_ids'] = shards['sharded_ids']
    return res


def _postprocess(all_features, true_ids, dataset_size, subset_size, nclasses, class_matches):
    all_features, labels = postprocess_features(all_features)
    return all_features, true_ids, dataset_size, subset_size, nclasses, labels, class_matches


def postprocess_features(all_features):
    all_features, labels = remove_labels(all_features)
    all_features = {view_key: [torch.Tensor(features) for features in view]
                    for view_key, view in all_features.items()}
    return all_features, labels


def postprocess_train(all_features, sample_level):
    all_features = {k: [postprocess_train_row(row, sample_level) for row in class_dt] for k, class_dt in all_features.items()}
    return all_features


def postprocess_train_row(row, sample_level):
    res = {}
    res['features'] = {k: torch.Tensor(v['features']) for k, v in row.items()}
    vids_dt = {k: v['vid'] for k, v in row.items()}
    vids = list(vids_dt.values())
    if sample_level:
        assert all(vid == vids[0] for vid in vids), "vid mismatch: {}".format(vids_dt)
    res['vid'] = vids[0]
    return res
