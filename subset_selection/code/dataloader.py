import time
import datetime
from functools import partial
from itertools import chain
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
# import concurrent.futures

from braceexpand import braceexpand
import numpy as np

from utils import load_json, load_pickle
from multiprocess import multiprocess


def format_row(x):
    i, row = x
    res = {}
    filename = row['filename']
    shard_name = row['shard_name']
    for feature_name in ['audio_assignments', 'video_assignments']:
        features = row[feature_name]
        for feature in features:
            model = feature['model_key']
            array = feature['array']
            if isinstance(array, dict):
                for layer in array.keys():
                    res[(model, layer)] = array[layer]
            elif isinstance(array, list):
                for i in range(array):
                    layer = "layer_{}".format(i)
                    res[(model, layer)] = array[layer]
            else:
                res[(model, 'model')] = array[layer]
    return i, (filename, shard_name, res)


def _format_assignment(x, clustering_types):
    i, row = x
    return i, [row[k] for k in clustering_types]


def format_assignments(assignments, num_workers=1, log_every=1000, verbose=False):
    # list(dict(clustering_type: clustering))
    clustering_types = sorted(list(assignments[0].keys()))
    format_assignment = partial(_format_assignment, clustering_types=clustering_types)
    assignments = list(multiprocess(format_assignment, assignments, num_workers,
                                    'formatting assignments samples',
                                    log_every=log_every,
                                    verbose=verbose))
    assignments = np.array(assignments)  # V x D
    return assignments, clustering_types


def preprocess(data, num_workers=1, log_every=1000, verbose=False):
    if verbose:
        print("preprocessing")
        print("formatting rows")
    filenames, shard_names, assignments = zip(*multiprocess(format_row, data, num_workers,
                                                            'formatting rows samples',
                                                            log_every=log_every,
                                                            verbose=verbose))
    if verbose:
        print("formatting assignments")
    assignments, clustering_types = format_assignments(assignments, num_workers,
                                                       log_every=log_every,
                                                       verbose=verbose)
    return assignments, shard_names, filenames, clustering_types


def load_partitions(shards_dir):
    # prefer newer logs
    log_paths = sorted(list(Path(shards_dir).glob('log_*.json')),
                       key=lambda x: str(x).split('.')[-2].split('_')[-1])
    partitions = {}
    for i, log_path in enumerate(log_paths):
        print("loading partition {}/{}".format(i, len(log_paths)))
        log = load_json(log_path)
        for shard in log['shards']:
            partitions[shard] = i
    # partitions = group(partitions)  # dict(i: set('shard_000000', ...))
    return partitions


def load_shard(shard_path):
    shard = load_pickle(shard_path)
    return shard_path, shard


def load_shards(shard_paths, num_workers, log_every=1000, verbose=False):
    start = time.time()
    if num_workers > 1:
        if verbose:
            print("parallel loading")
        '''
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(load_shard, p) for p in shard_paths]
            shards = [f.result() for f in futures]
        shards = dict(zip(shard_paths, shards))
        elasped = time.time() - start
        print("shards loading took {}".format(elasped))
        '''
        with Pool(num_workers) as p:
            shards = {}
            count = 0
            chunksize = max(1, len(shard_paths) // (num_workers))
            for shard_path, shard in p.imap_unordered(load_shard, shard_paths,
                                                      chunksize=chunksize):
                shards[shard_path] = shard
                count += 1
                if verbose:
                    if (count + 1) % log_every == 0:
                        elasped = time.time() - start
                        elasped = str(datetime.timedelta(seconds=elasped))
                        print("{}/{} clustering shards loaded (elasped: {})".format(count, len(shard_paths), elasped))
            '''
            shards = p.map(load_shard, shard_paths, chunksize=chunksize)
            shards = dict(zip(shard_paths, shards))
            elasped = time.time() - start
            print("shards loading took {}".format(elasped))
            '''
    else:
        if verbose:
            print("sequential loading")
        out = []
        count = 0
        for i, x in enumerate(shard_paths):
            shard_path, res = load_shard(x)
            out.append(res)
            count += 1
            if verbose:
                if (count + 1) % log_every == 0:
                    elasped = time.time() - start
                    elasped = str(datetime.timedelta(seconds=elasped))
                    print("{}/{} clustering shards loaded (elasped: {})".format(count, len(shard_paths), elasped))
        shards = dict(zip(shard_paths, out))
    return shards


def calc_lengths(li, num_workers):
    # return multiprocess(calc_length, li, num_workers, 'calc_len')
    # somehow parallelizing this part takes forever
    return [len(x) for x in li]


def calc_length(x):
    i, li = x
    return i, len(li)


def load_data(shard_paths, metas_path, num_workers=1, verbose=False,
              log_every=1000):
    if verbose:
        print("loading data")
    if not isinstance(shard_paths, list):
        shard_paths = list(braceexpand(str(shard_paths)))
    partitions = load_partitions(Path(shard_paths[0]).parent)
    shard_paths = [Path(shard_path) for shard_path in shard_paths]
    shard_paths = sorted([shard_path for shard_path in shard_paths if shard_path.is_file()])
    partitioned_data = defaultdict(list)
    shard_sizes = defaultdict(int)
    num_shards = defaultdict(int)
    no_logs = 0
    shards = load_shards(shard_paths, num_workers, log_every=log_every, verbose=verbose)

    partition_ids = []
    for shard_path in shard_paths:
        shard_name = shard_path.stem
        if shard_name in partitions:
            i = partitions[shard_name]
        else:
            i = -1
            no_logs += 1
        partition_ids.append(i)

    if len(set(partition_ids)) == 1:
        # single partition
        i = partition_ids[0]
        num_shards[i] = len(shard_paths)
        shards_list = list(shards.values())
        shard_sizes[i] = sum(calc_lengths(shards_list, num_workers))
        partitioned_data[i] = list(chain(*shards_list))
    else:
        for shard_path, shard in shards.items():
            shard_name = shard_path.stem
            if shard_name in partitions:
                i = partitions[shard_name]
            else:
                i = -1
                no_logs += 1
            num_shards[i] += 1
            shard_sizes[i] += len(shard)
            partitioned_data[i] += shard
    if verbose:
        print("num_shards: {} (clustering_partitions: {})".format(sum(num_shards.values()), dict(num_shards)))
        print("dataset_size: {} (clustering_partitions: {})".format(sum(shard_sizes.values()), dict(shard_sizes)))
        if no_logs > 0:
            print("num_shards without log files: {}".format(no_logs))
            print("shards with no log files will be processed as an independent set")
    metas = load_metas(shard_paths, metas_path, num_workers, verbose,
                       log_every=log_every)
    return partitioned_data, metas


def load_metas(shard_paths, metas_path, num_workers=1, verbose=False,
               log_every=1000):
    if verbose:
        print("loading metadata")
    start = time.time()
    metas_path = Path(metas_path)
    meta_paths = [metas_path / "{}.json".format(shard_path.stem) for
                  shard_path in shard_paths]
    if num_workers > 1:
        if verbose:
            print("parallel loading")
        metas = {}
        with Pool(num_workers) as p:
            count = 0
            chunksize = max(1, len(meta_paths) // (num_workers))
            for meta_path, meta in p.imap_unordered(load_meta, meta_paths,
                                                    chunksize=chunksize):
                metas[meta_path] = meta
                count += 1
                if verbose:
                    if (count + 1) % log_every == 0:
                        elasped = time.time() - start
                        elasped = str(datetime.timedelta(seconds=elasped))
                        print("{}/{} metadata shards loaded (elasped: {})".format(count, len(meta_paths), elasped))
    else:
        if verbose:
            print("sequential loading")
        out = []
        count = 0
        for i, x in enumerate(meta_paths):
            meta_path, res = load_meta(x)
            out.append(res)
            count += 1
            if verbose:
                if (count + 1) % log_every == 0:
                    elasped = time.time() - start
                    elasped = str(datetime.timedelta(seconds=elasped))
                    print("{}/{} metadata shards loaded (elasped: {})".format(count, len(meta_paths), elasped))
        metas = dict(zip(meta_paths, out))

    metas = {k.stem: v for k, v in metas.items() if v is not None}
    return metas


def load_meta(meta_path):
    meta = None
    if meta_path.is_file():
        meta = load_json(meta_path)
        meta = {Path(row['filename']).stem: row for row in meta}
    return meta_path, meta
