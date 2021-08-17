from functools import partial

from munch import Munch

import torch
from slowfast.utils.parser import load_config as load_slowfast_config
from slowfast.models.build import MODEL_REGISTRY
from slowfast.utils.checkpoint import load_checkpoint

from utils import (
    load_json, dump_json,
    load_pickle, dump_pickle,
    read_url, load_with_cache,
    download_file
)


def set_cache_dir(cache_dir):
    cache_dir = cache_dir / 'feature_extractors' / 'slowfast'
    cache_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    return cache_dir


def load_model(config, choices, cache_dir, device='cuda'):
    assert config in choices, f"no SlowFast backbone named {config}"
    config_name = f'{config}.yaml'.replace('/', '_')
    config_path = cache_dir / config_name
    if not config_path.is_file():
        slowfast_url = 'https://raw.githubusercontent.com/facebookresearch/SlowFast/master'
        config_url = f'{slowfast_url}/configs/{config}.yaml'
        download_file(config_url, config_path)
    args = build_slowfast_args(config_path)
    cfg = load_slowfast_config(args)
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)
    ckpt_path = load_weights(config, choices, cache_dir)
    convert_from_caffe2 = cfg.TEST.CHECKPOINT_TYPE == "caffe2"
    if config == 'Kinetics/c2/SLOWFAST_8x8_R50':
        # The checkpoint files use caffe2 without telling us so
        convert_from_caffe2 = True

    epoch = load_checkpoint(str(ckpt_path), model,
                            data_parallel=False,
                            convert_from_caffe2=convert_from_caffe2)
    if not convert_from_caffe2:
        assert epoch > 0, 'SlowFast ckpt not loaded!'
    model = model.to(device)
    model.eval()
    return model, cfg


def build_slowfast_args(config_path):
    args = {
        'shard_id': 0,
        'num_shards': 1,
        'init_method': 'tcp://localhost:9999',
        'cfg_file': config_path,
        'opts': None
    }
    return Munch(args)


def load_weights(config, choices, cache_dir):
    cache_path = cache_dir / f"{config.replace('/','_')}.pkl"
    if not cache_path.is_file():
        url = choices[config]
        download_file(url=url, out=cache_path)
    return cache_path


def load_model_options(cache_dir):
    cache_path = cache_dir / 'choices.json'
    return load_with_cache(cache_path, get_model_zoo)


def get_model_zoo():
    model_zoo_url = 'https://raw.githubusercontent.com/facebookresearch/SlowFast/master/MODEL_ZOO.md'
    model_zoo = read_url(model_zoo_url)
    data = parse_model_zoo(model_zoo)
    return data


def parse_model_zoo(model_zoo):
    reading_models = 0
    data = {}
    ckpt_num = -1
    config_num = -1
    title = ''

    def split_line(line):
        line = line.split('|')
        line = [element.strip() for element in line]
        line = [element for element in line if len(element) > 0]
        return line

    for line in model_zoo:
        line = line.strip()
        if reading_models == 2:
            if len(line) == 0:
                reading_models = 0
            else:
                line = split_line(line)
                if config_num < 0:
                    model_ckpt = line[ckpt_num]
                    model_ckpt = model_ckpt[model_ckpt.find('https://'): -1]
                    model_config = model_ckpt.split('/')[-1].split('.')[0]
                    if title is not None:
                        model_config = f'{title}/c2/{model_config}'
                    else:
                        model_config = None
                else:
                    model_config = line[config_num]
                    model_ckpt = line[ckpt_num]
                    model_config = model_config.strip()
                    model_ckpt = model_ckpt[model_ckpt.find('https://'): -1]

                if model_ckpt and model_config:
                    data[model_config] = model_ckpt
        elif reading_models == 0:
            if line.startswith('| architecture |'):
                line = split_line(line)
                ckpt_num = [i for i, v in enumerate(line) if v == 'model']
                ckpt_num = ckpt_num[0] if len(ckpt_num) > 0 else -1
                config_num = [i for i, v in enumerate(line) if v == 'config']
                config_num = config_num[0] if len(config_num) > 0 else -1
                title = 'AVA' if 'AVA version' in line else None
                reading_models = 1
        elif reading_models == 1:
            reading_models = 2
    return data
