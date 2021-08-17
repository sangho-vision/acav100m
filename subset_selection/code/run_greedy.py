import random
import time
import datetime

from pairing import get_cluster_pairing
from measures import get_measure


def _run_greedy(
    args,
    assignments,
    clustering_types,
    subset_size,
    subset_ratio,
    measure_name='mi',
    cluster_pairing='combination',
    shuffle_candidates=True,
    verbose=False
):
    ncentroids = assignments.max() + 1
    dataset_size = assignments.shape[0]
    if subset_size is None:
        subset_size = round(subset_ratio * dataset_size)
    if verbose:
        print("extracting {} samples from {} total datapoints".format(subset_size, dataset_size))
    clustering_combinations = get_cluster_pairing(clustering_types, cluster_pairing)

    batch_size = min(args.batch.batch_size, dataset_size - 1)
    selection_size = min(args.batch.selection_size, batch_size)

    measure = get_measure(measure_name)(assignments, ncentroids=ncentroids,
                                        batch_size=batch_size,
                                        selection_size=selection_size,
                                        device=args.computation.device,
                                        keep_unselected=args.batch.keep_unselected)

    candidates = list(range(dataset_size))
    if shuffle_candidates:
        print("shuffling candidates")
        random.shuffle(candidates)

    # start with singleton
    start_indices = [candidates[0]]
    candidates = candidates[1:]

    if verbose:
        print("initializing")
    measure.init(clustering_combinations, candidates)
    if verbose:
        print("done initialization")
    S, GAIN, timelapse, LOOKUPS = measure.run_greedy(subset_size, start_indices, None, verbose=verbose,
                                                     log_every=args.log_every, log_times=args.log_times,
                                                     node_rank=args.node_rank, pid=args.parent_pid)
    return S, GAIN, timelapse


def run_greedy(
    args,
    assignments,
    shard_names,
    filenames,
    clustering_types,
    subset_size,
    subset_ratio,
    measure_name='mi',
    cluster_pairing='combination',
    shuffle_candidates=True,
    verbose=False
):
    S, GAIN, timelapse = _run_greedy(args, assignments, clustering_types, subset_size, subset_ratio,
                                     measure_name, cluster_pairing, shuffle_candidates, verbose)
    S = sorted(list(S))
    data = [{'filename': filenames[s], 'shard_name': shard_names[s]} for s in S]
    return data
