from pathlib import Path

from munch import DefaultMunch

import torch

from config import defaults
from utils import get_run_id, get_run_info


def get_args(**kwargs):
    args = defaults
    args = update_args(args, kwargs)
    args = process_paths(args)
    args = objectify(args)
    if args.debug:
        torch.multiprocessing.set_sharing_strategy('file_system')  # bypass shm limit
    if 'computation.num_gpus' not in kwargs and args.computation.num_gpus is None:
        args.computation.num_gpus = torch.cuda.device_count()
    args.run_info = get_run_info()
    args.run_id = get_run_id(args.run_info)
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
    root = Path(args['root']).resolve()
    args['root'] = root
    suffixes = ['_file', '_dir']

    def _recurse(args, root):
        if 'path' in args:
            # root = root / args['path']
            # args['path'] = root
            args['path'] = Path(args['path']).resolve()
        for k, v in args.items():
            for suffix in suffixes:
                if k.endswith(suffix) and v is not None:
                    args[k] = root / v
                    break
            if isinstance(v, dict):
                args[k] = _recurse(v, root=root)
        return args

    args = _recurse(args, root)
    return args


def objectify(args):

    def _recurse(dt):
        for k, v in dt.items():
            if isinstance(v, dict):
                dt[k] = _recurse(v)
        dt = DefaultMunch(None, dt)
        return dt

    return _recurse(args)
