import os, contextlib
import pickle
from functools import reduce
from collections import defaultdict


def merge_dicts(li_of_dt):
    if len(li_of_dt) == 0:
        return {}
    res = {}
    for key in li_of_dt[0].keys():
        res[key] = flatten_dt([v[key] for v in li_of_dt])
    return res


def exchange_levels(dt):
    # keys = peek(dt).keys()
    res = defaultdict(dict)
    for k1, lv1 in dt.items():
        for k2, lv2 in lv1.items():
            res[k2][k1] = lv2
    return dict(res)


def flatten_dt(dt):
    if isinstance(dt, dict):
        return reduce(lambda x, y: {**x, **y}, dt.values())
    else:
        return reduce(lambda x, y: {**x, **y}, dt)


def peek(dt):
    keys = list(dt.keys())
    return dt[keys[0]]


def load_pickle(x):
    with open(x, 'rb') as f:
        x = pickle.load(f)
    return x


def dump_pickle(data, path):
    with open(str(path), 'wb') as f:
        pickle.dump(data, f)


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                func(*a, **ka)
        return wrapper


def merge_dataset_model_name(dataset, model):
    return "{}-{}".format(dataset, model)


def split_dataset_model_name(key):
    names = key.split('-')
    return names[0], '-'.join(names[1:])


def split_dataset_model_names(keys):
    keys = [split_dataset_model_name(key) for key in keys]
    res = defaultdict(list)
    for dataset, model in keys:
        res[dataset].append(model)
    return dict(res)
