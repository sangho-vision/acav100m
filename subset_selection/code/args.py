import os
import sys
from pathlib import Path

import torch
from munch import DefaultMunch

from config import defaults


def get_args(**kwargs):
    args = defaults
    args = update_args(args, kwargs)
    args = process_paths(args)
    args = objectify(args)

    args.computation.device = 'cpu'
    if args.computation.use_gpu:
        if torch.cuda.device_count() > 0:
            print("using gpu")
            args.computation.device = 'cuda'
        else:
            print("no gpu available, defaulting to cpu")

    if args.computation.num_gpus is None:
        args.computation.num_gpus = sys.maxsize
    args.computation.num_gpus = min(
        args.computation.num_gpus,
        torch.cuda.device_count()
    )

    print("args:")
    print(args)
    return args


def update_args(args, *additional_args):
    for new_args in additional_args:
        args = _update_args(args, new_args)
    return args


def _update_args(args, new_args):

    def _recurse(keys, args, v):
        if len(keys) > 0:
            current, keys = keys[0], keys[1:]
            dt = args[current] if current in args else {}
            args[current] = _recurse(keys, dt, v)
            return args
        else:
            return v

    for k, v in new_args.items():
        keys = k.split('.')
        args = _recurse(keys, args, v)

    return args


def process_paths(args):
    suffixes = ['_file', '_dir']

    def _recurse(args):
        if 'path' in args and args['path'] is not None:
            args['path'] = Path(args['path']).resolve()
        for k, v in args.items():
            for suffix in suffixes:
                if k.endswith(suffix) and v is not None:
                    args[k] = Path(v).resolve()
                    break
            if isinstance(v, dict):
                args[k] = _recurse(v)
        return args

    args = _recurse(args)
    return args


def objectify(args):

    def _recurse(dt):
        for k, v in dt.items():
            if isinstance(v, dict):
                dt[k] = _recurse(v)
        dt = DefaultMunch(None, dt)
        return dt

    return _recurse(args)
