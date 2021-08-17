import pickle
import warnings
import random
from pathlib import Path

import torch
import webdataset as wds
from torch.utils.data import IterableDataset, DataLoader

from utils import identity, get_num_workers, to_str, load_pickle
from mps import distributed as du
from .shuffle import shuffle
from .pipeline import pipeline
from .shards import get_shards_size, get_shards_path


def get_clustering_dataloader(args, drop_last=False, shuffle=False, is_train=True):
    # is_train: True when training centroids, False when assigning cluster labels
    dataset, num_workers = get_dataset(args, is_train=is_train)
    if isinstance(args.computation.num_gpus, int):
        world_size = min(du.get_world_size(), args.computation.num_gpus)
    else:
        world_size = du.get_world_size()

    batch_size = int(args.data.batch_size / world_size)

    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=False,  # shuffling in dataloader is meaningless for webdataset
        num_workers=num_workers,
        collate_fn=collate_features,
        drop_last=drop_last)
    return dataloader


def get_dataset(args, is_train=True):
    shards_path, rest = get_shards_path(args, suffix='.pkl', f=get_shards_size, is_train=is_train)
    data = FeatureDataset(args, shards_path, rest['all_shards_path'], is_train=is_train)
    if isinstance(args.computation.num_gpus, int):
        world_size = min(du.get_world_size(), args.computation.num_gpus)
    else:
        world_size = du.get_world_size()

    batch_size = int(args.data.batch_size / world_size)
    num_workers = data.num_workers
    effective_num_workers = 1 if num_workers == 0 else num_workers

    shards_size_dt = rest['shards_size_dt']
    shards_size = [shards_size_dt[Path(p).stem] for p in rest['all_shards_path']]
    length = du.get_length(
        shards_size, batch_size, num_workers, is_train=is_train,
    )
    print("(node {}) dataset size: {}".format(
        du.get_rank(),
        sum(du.node_selection(
            shards_size, du.get_rank(),
            total=world_size, is_train=is_train, no_str_ok=True))))
    if du.get_rank() == 0:
        print("dataset length: {}".format(length))
    nominal = length * effective_num_workers
    data = wds.ResizedDataset(
        data,
        length,
        nominal,
    )
    return data, num_workers


def get_layer(array, layer):
    if isinstance(array, dict):
        return array[layer]
    elif isinstance(array, list):
        i = int(layer.split('_')[-1])
        return array[i]
    raise ValueError('feature array is not a dict nor a list!')


def collate_features(batch):
    feature_names = ['video_features', 'audio_features']
    pivot = batch[0]
    res = {}
    for key in pivot.keys():
        if key in feature_names:
            for i, _ in enumerate(pivot[key]):
                if isinstance(pivot[key][i], dict):  # layer extractor
                    pivot_array = pivot[key][i]['array']
                    feature = {}
                    if isinstance(pivot_array, dict):
                        layer_keys = list(pivot_array.keys())
                    elif isinstance(pivot_array, list):
                        layer_keys = [f'layer_{i}' for i in range(len(pivot_array))]
                    else:
                        raise ValueError('feature array is not a dict nor a list!')
                    for layer in layer_keys:
                        layer_feature = []
                        for row in batch:
                            try:
                                layer_feature.append(torch.from_numpy(get_layer(row[key][i]['array'], layer)))
                            except Exception as e:
                                print(f"{row['shard_name']} shard error: {e}")
                                raise Exception
                        layer_feature = torch.stack(layer_feature, dim=0)
                        feature[layer] = layer_feature
                else:
                    feature = [torch.from_numpy(row[key][i]['array']) for row in batch]
                    feature = torch.stack(feature, dim=0)
                model_key = (pivot[key][i]['extractor_name'], pivot[key][i]['dataset'])
                model_key = '/'.join(model_key)
                res[model_key] = feature
        else:
            res[key] = [row[key] for row in batch]
    res['idx'] = [Path(row['filename']).stem for row in batch]
    return res


class FeatureDataset(IterableDataset):
    def __init__(self, args, shards_path, all_shards_path,
                 node_selection=identity, shard_shuffle=identity, is_train=True):
        # is_train: True when training centroids, False when assigning cluster labels
        # We need the list of paths to all input shards
        # (after discarding if args.computation.discard_shards is set)
        # Here, I'll refer to it as `all_shards_path`
        self.shards_path = shards_path
        self.all_shards_path = all_shards_path
        if is_train:
            if isinstance(args.computation.num_gpus, int):
                world_size = min(du.get_world_size(), args.computation.num_gpus)
            else:
                world_size = du.get_world_size()
            num_shards = [
                len(du.node_selection(all_shards_path, i, total=world_size, is_train=is_train))
                for i in range(world_size)
            ]
            self.num_workers = min(
                [args.computation.num_workers] + num_shards
            )
        else:
            # Here, self.shards_path is the list of paths to shards allocated to current node (gpu)
            # (after discarding if args.computation.discard_shards is set)
            self.num_workers, _ = get_num_workers(
                args.computation.num_workers, len(self.shards_path),
            )

        out_str = "#Workers of Feature Extraction Dataset"
        out_str += f" (train={is_train}, node={du.get_rank()})"
        out_str += f": {self.num_workers}"
        print(out_str)
        self.node_selection = node_selection
        self.shard_shuffle = shard_shuffle

        self.pipeline = []

    def shard_fn(self):
        urls = self.shards_path
        urls = self.node_selection(urls)
        urls = worker_urls(urls)
        urls = self.shard_shuffle(urls)
        return urls

    def samples(self, urls):
        if isinstance(urls, str):
            urls = [urls]
        assert isinstance(urls, list)
        source = self.raw_samples(urls)
        return pipeline(source, *self.pipeline)

    def raw_samples(self, urls):
        for url in urls:
            url = Path(url)
            try:
                try:
                    pkl = load_pickle(url)
                except EOFError as e:
                    print(e)
                    print('EOFError in shard loading: {}'.format(Path(url.stem)))
                    continue
                for feature in pkl:
                    yield feature
            except Exception as e:
                print(e)
                print('Exception in shard loading: {}'.format(Path(url.stem)))
                continue

    def __iter__(self):
        urls = self.shard_fn()
        return self.samples(urls)

    def shuffle(self, size, rng=None, **kw):
        """Shuffle the data."""
        if size == 0:
            return self
        if rng is None:
            rng = random.Random()
        self.rng = rng
        self.shard_shuffle = Shuffler(rng)
        self.pipeline.append(shuffle(size, rng=rng, **kw))
        return self


class Shuffler:
    """Make a shuffle function (avoid nesting for pickle)."""

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, lst):
        lst = list(lst)
        self.rng.shuffle(lst)
        return lst


def worker_urls(urls):
    """Selects a subset of urls based on Torch get_worker_info.
    Used as a shard selection function in Dataset."""
    import torch

    assert isinstance(urls, list)
    assert isinstance(urls[0], str)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        wid = worker_info.id
        num_workers = worker_info.num_workers
        if wid == 0 and len(urls) < num_workers:
            warnings.warn("num_workers {} > num_shards {}".format(num_workers, len(urls)))
        return urls[wid::num_workers]
    else:
        return urls
