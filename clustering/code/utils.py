import os
import json
import pickle
import urllib
import platform
import time
import datetime
from subprocess import check_output, Popen, PIPE
from pathlib import Path

import wget
import torch
import numpy as np
import braceexpand


def filter_models(args):
    model_names = args.models  # preserve order
    data_name = args.data.media.path.stem
    if data_name in args.data.types and args.data.types[data_name] == 'audio_only':
        model_names = [name for name in model_names if name in args.model_types.audio]

    return model_names


def to_str(li):
    return [str(x) for x in li]


def get_data_cache_path(args):
    name = Path(args.data.path).name
    return args.data.output.path, name


def get_num_workers(num_workers, num_shards):
    # Here, num shards is the number of shards allocated to current node (gpu)
    if num_workers > num_shards:
        num_workers = num_shards
    effective_num_workers = 1 if num_workers == 0 else num_workers
    return num_workers, effective_num_workers


def get_idx(filename):
    return Path(filename).stem


def get_tar_size(filename):
    file_list = Popen(["tar", "-tvf", filename], stdout=PIPE)
    tar_size = check_output(["wc", "-l"], stdin=file_list.stdout).strip().decode('utf-8')
    assert tar_size.isdigit(), f"invalid return from tar_size computation: {tar_size}"
    tar_size = int(tar_size)
    return tar_size


def get_run_info():
    info = {}
    info['hostname'] = platform.uname()[1]
    info['pid'] = os.getpid()
    info['timestamp'] = int(time.time())
    info['time'] = str(datetime.datetime.now())
    return info


def get_run_id(run_info=None):
    if run_info is None:
        run_info = get_run_info()
    keys = ['hostname', 'pid', 'timestamp']
    val = [str(run_info[key]) for key in keys if key in run_info]
    return '_'.join(val)


def identity(x):
    return x


def to_brace(li):
    if len(li) == 0:
        return ''
    elif len(li) == 1:
        return li[0]
    else:
        return '{' + ','.join(li) + '}'


def identity(x):
    return x


def get_tensor_size(t):
    if isinstance(t, np.ndarray):
        return t.size
    elif torch.is_tensor(t):
        return t.nelement()
    return None


def to_device(it, device):
    if torch.is_tensor(it):
        return it.to(device)
    elif isinstance(it, dict):
        for k, v in it.items():
            it[k] = to_device(v, device)
    elif isinstance(it, list):
        it = [to_device(v, device) for v in it]
    return it


def ensure_parents(path):
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(str(path), 'r') as f:
        data = json.load(f)
    return data


def dump_json(data, path, indent=4):
    ensure_parents(path)
    with open(str(path), 'w') as f:
        json.dump(data, f, indent=indent)
    return


def load_pickle(path):
    try:
        with open(str(path), 'rb') as f:
            data = pickle.load(f)
    except EOFError as e:
        raise EOFError("Ran out of Input: (file path: {}), (msg: {})".format(path, e))
    return data


def dump_pickle(data, path):
    ensure_parents(path)
    with open(str(path), 'wb') as f:
        pickle.dump(data, f)
    return


def read_url(url):
    with urllib.request.urlopen(url) as f:
        lines = []
        for line in f:
            line = line.decode('utf-8')
            lines.append(line)
    return lines


def download_file(url, out=None):
    url = str(url)
    if out is not None:
        out = Path(out)
        if out.is_file():
            print(f"File already exists: {str(out)}")
        out = str(out)
    wget.download(url, out=out)


def load_with_cache(cache_path, get_file,
                    load_file=load_json,
                    dump_file=dump_json,
                    dumped=False):
    cache_path = Path(cache_path)
    if cache_path.is_file():
        return load_file(cache_path)
    else:
        data = get_file()
        if not dumped:
            dump_file(data, cache_path)
        return data


def dol_to_lod(dol):
    keys = sorted(list(dol.keys()))
    ls = [dol[k] for k in keys]
    return [dict(zip(keys, row)) for row in zip(*ls)]
