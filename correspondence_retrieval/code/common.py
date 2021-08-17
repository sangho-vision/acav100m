import pickle
import time
from pathlib import Path

from tqdm import tqdm


save_keys = [
    'data_name', 'extract_each_layer',
    'nsamples_per_class',
    'clustering_func_type', 'measure',
    'cluster_pairing', 'weight_type',
    'ncentroids',
    'num_shards', 'share_clustering',
    'batch_size', 'selection_size'
]


def get_name(args):
    keys = save_keys
    return {key: args.get(key, None) for key in keys}


def get_cache_path(args, out_path):
    dir_args = get_name(args)
    path = Path(out_path)
    path.mkdir(exist_ok=True, parents=True)
    dir_name = ''
    dir_keys = save_keys
    for name in dir_keys:
        val = dir_args.pop(name)
        name = "{}_{}".format(name, val)
        dir_name = "{}/{}".format(dir_name, name)
    dir_name = dir_name[1:]
    path = path / dir_name
    path.mkdir(exist_ok=True, parents=True)
    keys = sorted(list(dir_args.keys()))
    name = "_".join("{}_{}".format(key, dir_args[key]) for key in keys)
    name = "clustering_{}.pkl".format(name)
    path = path / name
    return path


def save_output(out_path, stats, additional_data, i, expr_name, args, verbose=True):
    dir_args = get_name(args)
    path = Path(out_path)
    path.mkdir(exist_ok=True)
    dir_name = ''
    dir_keys = save_keys
    for name in dir_keys:
        val = dir_args.pop(name)
        name = "{}_{}".format(name, val)
        dir_name = "{}/{}".format(dir_name, name)
    dir_name = dir_name[1:]

    path = path / dir_name
    path.mkdir(exist_ok=True, parents=True)
    keys = sorted(list(dir_args.keys()))
    name = "_".join("{}_{}".format(key, dir_args[key]) for key in keys)
    if expr_name is not None:
        name = "name_{}_{}".format(expr_name, name)
    if i is not None:
        name = "expr_{}_{}".format(i, name)

    if 'log_keys' in args:
        log_key_name = "_".join("{}_{}".format(key, args[key]) for key in args['log_keys'])
        name = "{}_{}".format(name, log_key_name)

    name = "{}_{}".format(name, time.time())
    name = "{}.pkl".format(name)
    path = path / name
    if verbose:
        tqdm.write("Saving to {}".format(path))
    data = {
        'args': args,
        'stats': stats,
        **additional_data
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def get_stats(gold_standard, S):
    precision = len(set(gold_standard) & set(S)) / len(set(S))
    recall = len(set(gold_standard) & set(S)) / len(set(gold_standard))
    if precision + recall != 0.0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1
