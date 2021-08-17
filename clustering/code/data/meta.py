import json
from pathlib import Path

from tqdm import tqdm

from mps import distributed as du
from utils import load_pickle, dump_pickle


def load_metadata(args, shard_paths):
    if args.data.meta.path is not None:
        meta_path = Path(args.data.meta.path)
        cache_path = meta_path / 'meta_cache.pkl'
        if cache_path.is_file():
            shards_size_dt = load_pickle(cache_path)
            to_load = [p for p in shard_paths if Path(p).stem not in shards_size_dt.keys()]
            if len(to_load) > 0:
                _, _, shards_size_dt_upd = _load_metadata(args, to_load)
                shards_size_dt = {**shards_size_dt, **shards_size_dt_upd}
                dump_pickle(shards_size_dt, cache_path)
        else:
            _, _, shards_size_dt = _load_metadata(args, shard_paths)
            dump_pickle(shards_size_dt, cache_path)
    else:
        _, _, shards_size_dt = _load_metadata(args, shard_paths)
    return shards_size_dt


def _load_metadata(args, shard_paths):
    shards_size = []
    shards_size_dt = {}
    metadata = {}
    pbar = shard_paths
    if du.get_rank() == 0:
        # only for one process
        print("loading metadata from json files")
        pbar = tqdm(shard_paths)
    for shard_path in pbar:
        shard_name = Path(shard_path).stem
        if args.data.meta.path is not None:
            meta_path = Path(args.data.meta.path)
            if meta_path.is_dir():
                meta_path = meta_path / "{}.json".format(shard_name)
        else:
            meta_path = Path(shard_path).parent / "{}.json".format(shard_name)
        if meta_path.is_file():
            with open(meta_path, 'r') as f:
                shard_file = json.load(f)
                count = len(shard_file)
                for line in shard_file:
                    idx = line['filename'].split('.')[0]
                    line['shard_size'] = count
                    line['shard_name'] = shard_name
                    metadata[idx] = line
            shards_size.append(count)
            shards_size_dt[shard_name] = count
    return metadata, shards_size, shards_size_dt
