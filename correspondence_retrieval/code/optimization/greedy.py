import time
import random

from tqdm import tqdm

import numpy as np


def greedy(
    measure,
    dataset_size,
    subset_size,
    start_indices,
    intermediate_target=None,
    clustering_combinations=None,
    verbose=True
):
    candidates = list(set(range(dataset_size)) - set(start_indices))
    random.shuffle(candidates)

    greedy_start_time = time.time()
    start_time = time.time()
    Q = [[c, None] for c in candidates]
    S = start_indices
    GAIN = []
    LOOKUPS = []
    timelapse = []
    agreed_dict = {}

    pbar = tqdm(range(len(start_indices), subset_size - 1), desc='greedy iter')
    for j in pbar:
        best_idx = -1
        best_measure = -np.inf

        for i in range(len(Q)):
            current = Q[i][0]
            current_measure, current_agreed_dict = measure(
                S + [current], clustering_combinations=clustering_combinations,
                agreed_dict=agreed_dict,
            )

            if current_measure > best_measure:
                best_measure = current_measure
                best_idx = i
                best_agreed_dict = current_agreed_dict

        agreed_dict = best_agreed_dict
        S.append(Q[best_idx][0])
        lookup = len(Q)
        LOOKUPS.append(lookup)
        GAIN.append(best_measure)
        timelapse.append(time.time() - start_time)
        del Q[best_idx]
        '''
        import copy
        c = copy.deepcopy(agreed_dict['count']['[0, 1]'])
        if c_prev is not None:
            print(c - c_prev)
        c_prev = c
        '''
        if intermediate_target is not None:
            precision = len(set(intermediate_target) & set(S)) / len(set(S))
            pbar.set_description("(LEN: {}, MEASURE: {}, PRECISION: {})".format(
                len(S), best_measure, precision))
        else:
            pbar.set_description("(LEN: {}, MEASURE: {})".format(len(S), best_measure))

    if verbose:
        tqdm.write("Time Consumed: {} seconds".format(time.time() - greedy_start_time))
    return (S, GAIN, timelapse, LOOKUPS)
