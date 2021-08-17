import time
import datetime
import json
from pathlib import Path
from functools import reduce
from multiprocessing import Pool

from mps import distributed as du
from multiprocess import multiprocess


def load_metadata(args, shard_paths):
    metadata, _, shards_size_dt = _load_metadata(args, shard_paths)
    return metadata, shards_size_dt


def _load_metadata(args, shard_paths):
    verbose = args.verbose and du.is_master_proc()
    shards_size = []
    shards_size_dt = {}
    metadata = {}
    if du.is_master_proc():
        # only for one process
        print("loading metadata from json files")
        pbar = shard_paths
    meta_paths = []
    all_meta_paths = []
    not_loaded = []
    for shard_path in shard_paths:
        shard_name = Path(shard_path).stem
        if args.data.meta.path is not None:
            meta_path = Path(args.data.meta.path)
            if meta_path.is_dir():
                meta_path = meta_path / "{}.json".format(shard_name)
        else:
            meta_path = Path(shard_path).parent / "{}.json".format(shard_name)
        all_meta_paths.append(meta_path)
        if meta_path.is_file():
            meta_paths.append(meta_path)
        else:
            print("no meta file named: {}".format(meta_path))

    metadata, shards_size, shards_size_dt = map_reduce(args, meta_paths, all_meta_paths)
    if verbose:
        print("done loading metadata")
    return metadata, shards_size, shards_size_dt


def map_reduce(args, meta_paths, all_meta_paths):
    num_workers = 1 if not args.computation.multiprocess_meta_loading else args.computation.num_workers
    verbose = args.verbose and du.is_master_proc()

    start = time.time()
    if verbose:
        print("multiprocessing metadata loading")
    res = multiprocess(load_shard, meta_paths, args.computation.num_workers,
                       granularity='shards', log_every=args.log_every, verbose=verbose)
    if verbose:
        elasped = time.time() - start
        elasped = str(datetime.timedelta(seconds=elasped))
        print("done multiprocessing {} shards (elasped: {})".format(len(res), elasped))

    if verbose:
        print("splitting multiprocess outputs")
    metadata, counts = zip(*res)
    '''
    if verbose:
        print("merging metadata dictionaries")
    metadata = reduce(lambda x, y: {**x, **y}, metadata)  # merge dicts
    '''
    if verbose:
        print("aligning shard_name as a key to metadata")
    shard_names, _ = zip(*counts)
    metadata = dict(zip(shard_names, metadata))
    if verbose:
        print("building shards_size_dt")
    shards_size_dt = dict(counts)

    if verbose:
        print("handling not_loaded if there is one")
    shard_names = [Path(meta_path).stem for meta_path in all_meta_paths]
    not_loaded = {k: 0 for k in (set(shard_names) - set(shards_size_dt.keys()))}
    if len(not_loaded) > 0:
        if verbose:
            print("handling {} not_loaded metadata files".format(len(not_loaded)))
        shards_size_dt = {**shards_size_dt, **not_loaded}
    else:
        if verbose:
            print("no not_loaded files")
    if verbose:
        print("converting shards_size_dt to shards_size list")
    shards_size = [shards_size_dt[name] for name in shard_names]
    return metadata, shards_size, shards_size_dt


def load_shard(meta_path):
    i, meta_path = meta_path
    shard_name = Path(meta_path).stem
    metadata = {}
    with open(meta_path, 'r') as f:
        shard_file = json.load(f)
        count = len(shard_file)
        for line in shard_file:
            idx = line['filename'].split('.')[0]
            line['shard_size'] = count
            line['shard_name'] = shard_name
            metadata[idx] = line
    return i, (metadata, (shard_name, count))


'''
def multiprocess_meta(data, num_workers=1, granularity='shards',
                 log_every=1000, verbose=False):
    start = time.time()
    if num_workers > 1:
        if verbose:
            print("parallel processing")
        metadata = {}
        shards_size_dt = {}
        with Pool(num_workers) as p:
            count = 0
            chunksize = max(1, len(data) // (num_workers))
            for i, res in p.imap_unordered(load_shard, enumerate(data), chunksize=chunksize):
                shard_metadata, (shard_name, shard_size) = res
                metadata = {**metadata, **shard_metadata}
                shards_size_dt[shard_name] = shard_size
                count += 1
                if verbose:
                    if (count + 1) % log_every == 0:
                        elasped = time.time() - start
                        elasped = str(datetime.timedelta(seconds=elasped))
                        print("{}/{} {} processed (elasped: {})".format(count, len(data), granularity, elasped))
    else:
        if verbose:
            print("sequential processing")
        out = []
        count = 0
        for i, x in enumerate(data):
            i, res = load_shard((i, x))
            out.append(res)
            count += 1
            if verbose:
                if (count + 1) % log_every == 0:
                    elasped = time.time() - start
                    elasped = str(datetime.timedelta(seconds=elasped))
                    print("{}/{} {} processed (elasped: {})".format(count, len(data), granularity, elasped))
        out = dict(enumerate(out))
    if verbose:
        print("sorting multiprocess outputs")
    out = [out[k] for k in sorted(list(out.keys()))]
    return metadata, shards_size_dt
'''
