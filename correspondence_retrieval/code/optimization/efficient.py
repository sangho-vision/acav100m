import time
import random

from tqdm import tqdm

import numpy as np


def efficient_greedy(
    measure,
    dataset_size,
    subset_size,
    start_indices,
    intermediate_target=None,
    clustering_combinations=None,
    celf_ratio=0,
    verbose=True
):
    candidates = list(set(range(dataset_size)) - set(start_indices))
    random.shuffle(candidates)

    if verbose:
        print("initializing")
    measure.init(clustering_combinations, candidates)
    if verbose:
        print("done initialization")
    return measure.run(subset_size, start_indices, intermediate_target, celf_ratio)
