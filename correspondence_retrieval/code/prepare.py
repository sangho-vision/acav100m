import random
from pathlib import Path

import torch
import numpy as np

from args import get_args
from image_pair_data import get_image_pair_data
from measures import EFFICIENT_MEASURES


def get_data(args):
    return get_image_pair_data(args)


def prepare(verbose=True, **kwargs):
    args = get_args(**kwargs)
    if args.data_name == 'kinetics_sounds':
        args.sample_level = True
    if args.measure.startswith('efficient_'):
        args.optimization_algorithm = 'efficient_greedy'
    elif args.measure in EFFICIENT_MEASURES:
        args.optimization_algorithm = 'efficient_greedy'
    if args.measure == 'contrastive':
        if args.train_ratio is None:
            args.train_ratio = 0.5
        args.use_gpu = True
    args.shuffle_datapoints = args.data_name not in args.sample_level_correspondence_data

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.device = 'cpu'
    if args.use_gpu:
        if torch.cuda.device_count() > 0:
            print("using gpu")
            args.device = 'cuda'
        else:
            print("no gpu available, defaulting to cpu")

    args.out_path = str(Path(args.out_root) / args.out_dir)
    Path(args.out_path).mkdir(exist_ok=True, parents=True)
    if verbose:
        print(dict(args))
    shards = get_data(args)
    return args, shards
