import os
import inspect
from pathlib import Path

import torch

from torch import nn

from inflection import underscore


# models should be initialized before loading data


model_dict = {}


def add_models():
    path = Path(os.path.dirname(__file__))

    for p in path.glob('*.py'):
        name = p.stem
        parent = p.parent.stem
        if name != "__init__":
            __import__(f"{parent}.{name}")
            module = eval(name)
            for member in dir(module):
                # Add to dict all nn.Module classes
                member = getattr(module, member)
                if hasattr(member, '__mro__') and \
                        nn.Module in inspect.getmro(member):
                    model_dict[underscore(str(member.__name__))] = member


def get_model_dict():
    if not model_dict:
        add_models()
    return model_dict


def get_model_class(model_name, args):
    if not model_dict:
        add_models()
    model_class = model_dict[model_name]
    if hasattr(model_class, 'args'):
        args = merge_args(args, model_class.args)
    return model_class, args


def init_model(model_name, args):
    model_class, args = get_model_class(model_name, args)
    return model_class(args), args


def merge_args(args, model_args):
    for k, v in model_args.items():
        if k not in args:
            args[k] = model_args[k]
    return args


def get_model(model_name, args):
    model, args = init_model(model_name, args)
    model = model.to(args.computation.device)
    '''
    if args.computation.num_gpus > 1 and args.task != "extraction":
        cur_device = torch.cuda.current_device()
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
    '''
    return model, args


def download_model_weights(args):
    model_names = sorted(args.models)
    for model_name in model_names:
        model_class, args = get_model_class(model_name, args)
        if hasattr(model_class, 'download'):
            print("downloading {} weights".format(model_name))
            model_class.download(args)
