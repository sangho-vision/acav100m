from itertools import combinations

import numpy as np

from utils import get_pbar
from dataloader import load_data, preprocess
from run_greedy import _run_greedy


def compare_measures(args):
    if not hasattr(args, 'measure_names') or args.measure_names is None:
        args.measure_names = ['mem_mi', 'mi']
    partitions, metas = load_data(args.data.path, args.data.meta.path, args.verbose)
    pbar = get_pbar(sorted(list(partitions.keys())), args.verbose, desc='partition')
    results = []
    for k in pbar:
        partition = partitions[k]
        samples_list = run_partition(args, partition, metas)
        results.append(samples_list)


def run_partition(args, data, metas):
    assignments, shard_names, filenames, clustering_types = preprocess(data)
    Ss = {}
    GAINs = {}
    timelapses = {}
    for measure_name in args.measure_names:
        S, GAIN, timelapse = _run_greedy(assignments, clustering_types,
                                args.subset.size,
                                args.subset.ratio,
                                measure_name=measure_name,
                                cluster_pairing=args.clustering.pairing,
                                shuffle_candidates=args.shuffle_candidates,
                                verbose=args.verbose)
        Ss[measure_name] = S
        GAINs[measure_name] = GAIN
        timelapses[measure_name] = timelapse

    for k1, k2 in combinations(list(Ss.keys()), 2):
        print(k1, 'vs.', k2)
        sames = np.array([int(v1 == v2) for v1, v2 in zip(Ss[k1], Ss[k2])])
        print('S equivalence: ', sames.mean())
        gain_diffs = np.array([abs(v1 - v2) for v1, v2 in zip(GAINs[k1], GAINs[k2])])
        print('GAIN diff mean: ', gain_diffs.mean())
        import ipdb; ipdb.set_trace()  # XXX DEBUG
    return Ss[args.measure_names[0]]
