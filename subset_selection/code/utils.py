import math
import json
import pickle
from itertools import groupby


def get_num_workers(num_workers, num_shards):
    # Here, num shards is the number of shards allocated to current node (gpu)
    if num_workers > num_shards:
        num_workers = num_shards
    effective_num_workers = 1 if num_workers == 0 else num_workers
    return num_workers, effective_num_workers


def get_pbar(iters, verbose, desc=None):
    return iters
    '''
    if verbose:
        return tqdm(iters, desc=desc)
    else:
    '''


def peek(dt):
    keys = list(dt.keys())
    return dt[keys[0]]


def to_str(li):
    return [str(x) for x in li]


def get_num_workers(num_workers, num_shards):
    # Here, num shards is the number of shards allocated to current node (gpu)
    if num_workers > num_shards:
        num_workers = num_shards
    effective_num_workers = 1 if num_workers == 0 else num_workers
    return num_workers, effective_num_workers


def identity(x):
    return x


def load_json(p):
    with open(p, 'r') as f:
        x = json.load(f)
    return x


def dump_json(x, p):
    with open(p, 'w') as f:
        json.dump(x, f)


def load_pickle(p):
    with open(p, 'rb') as f:
        x = pickle.load(f)
    return x


def dump_pickle(x, p):
    with open(p, 'wb') as f:
        pickle.dump(x, f)


def get_first(x):
    return x[0]


def group(x):
    x = list(groupby(sorted(x, key=get_first), get_first))
    x = {k: set(v) for k, v in x}
    return x


def get_chunks(li, n):
    # split into chunks with size n
    for i in range(0, len(li), n):
        yield li[i: i + n]


def split_chunks(li, m):
    # split into m chunks
    n = math.ceil(len(li) / m)
    return get_chunks(li, n)
