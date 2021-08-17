from pathlib import Path

from tqdm import tqdm

from clustering import run_clusterings, Clustering
from start_indices import get_start_indices
from optimization import optimize
from measures import get_measure
from cluster_pairing import get_cluster_pairing
from common import save_output, get_stats


def _run(args, features, true_ids, dataset_size, subset_size,
         nclasses, labels, class_matches, train_data=None,
         expr_name=None, clustering=None, if_save=True, verbose=True):
    if verbose:
        tqdm.write("Dataset Size: {}".format(dataset_size))
        tqdm.write("Target Ratio: {}".format(subset_size / dataset_size))

    if args.measure != 'contrastive':
        if clustering is None:  # metric learning measures do not require clusterings
            clustering = run_clusterings(args, features, args.ncentroids, args.kmeans_iters,
                                        args.clustering_func_type)
    else:
        clustering = features
    S = run_optimizations(args, clustering, true_ids,
                        expr_name=expr_name,
                        dataset_size=dataset_size, subset_size=subset_size, nclasses=nclasses,
                        ntargets_per_class=args.ntargets_per_class,
                        measure=args.measure, algorithm=args.optimization_algorithm,
                        get_intermediate_stats=True,
                        celf_ratio=args.celf_ratio,
                        cluster_pairing=args.cluster_pairing,
                        weight_type=args.get('weight_type', None),
                        train_data=train_data,
                        nexprs=args.nexprs, selection=args.start_indices_selection, all_args=args,
                        out_path=Path(args.out_path),
                        if_save=if_save, verbose=verbose)
    return S


def run_optimizations(*args, nexprs=10, all_args={}, selection='random_one_per_class', **kwargs):
    if 'random' in selection:
        for i in tqdm(range(nexprs), ncols=80):
            S = run_optimization(*args, i=i, all_args=all_args, selection=selection, **kwargs)
    else:
        S = run_optimization(*args, i=None, all_args=all_args, selection=selection, **kwargs)
    return S


def run_optimization(*args, i=None, expr_name=None, all_args={}, out_path=None, if_save=True, verbose=True, **kwargs):
    S, stats, additional_data = _run_optimization(*args, verbose=verbose, **kwargs)
    print_str = "precision: {:.3f} | recall: {:.3f} | f1: {:.3f}".format(
        stats['precision'][-1], stats['recall'][-1], stats['f1'][-1],
    )
    if i is not None:
        print_str = "{}-th expr ... {}".format(i, print_str)
    if verbose:
        tqdm.write(print_str)

    if if_save:
        print(all_args)
        save_output(out_path, stats, additional_data, i, expr_name, all_args, verbose)
    return S


def _run_optimization(args, clustering, gold_standard, dataset_size, subset_size, nclasses, ntargets_per_class, selection,
                      measure='custom', algorithm='celf', get_intermediate_stats=False,
                      cluster_pairing='combination', train_data=None,
                      weight_type=None, celf_ratio=0, verbose=True):
    if args.measure != 'contrastive':
        start_indices = get_start_indices(clustering, nclasses, ntargets_per_class, selection)
        if verbose:
            print("start_indices: {}".format(start_indices))
            print("num_(view * layer)s: {}".format(len(clustering)))
        clustering_keys = sorted(list(clustering.keys()))
        clustering_list = [clustering[key] for key in clustering_keys]
        clustering_combinations = get_cluster_pairing(clustering_keys, cluster_pairing,
                                                      weight_type)
        data = clustering_list
    else:
        start_indices = []  # do not use start_indices
        clustering_combinations = None
        clustering_list = None
        test = clustering  # hack..
        data = (train_data, test)

    if verbose:
        print("num_clustering_pairs: {}".format(len(clustering_combinations)))

    if celf_ratio > 0:
        assert algorithm == 'efficient_greedy', 'celf_ratio > 0 only works with efficient_greedy'

    measure = get_measure(args, data, measure=measure)
    intermediate_target = gold_standard if get_intermediate_stats else None
    S, GAIN, timelapse, LOOKUPS = optimize(
        measure,
        dataset_size,
        subset_size,
        start_indices,
        algorithm=algorithm,
        intermediate_target=intermediate_target,
        clustering_combinations=clustering_combinations,
        celf_ratio=celf_ratio,
        verbose=verbose
    )
    assert len(S) == len(set(S)), "duplicates in S: {} duplicates".format(len(S) - len(set(S)))

    stats = []
    for i in range(1, len(S) + 1):
        stats.append(get_stats(gold_standard, S[:i]))
    stats = list(zip(*stats))
    stats = dict(zip(['precision', 'recall', 'f1'], stats))
    stats['gain'] = GAIN
    stats['lookups'] = LOOKUPS

    print("precision: {}".format(stats['precision'][-1]))

    if clustering_list is not None and isinstance(clustering_list[0], Clustering):
        assignments = {k: [C.get_assignment(idx) for idx in start_indices] for k, C in clustering.items()}
        cluster_sizes = {k: {cen: len(v) for cen, v in C.cen2ind.items()} for k, C in clustering.items()}
    else:
        assignments = {k: [] for k, C in clustering.items()}
        cluster_sizes = {k: {} for k, C in clustering.items()}
    additional_data = {
        'start': {
            'indices': start_indices,
            'assignments': assignments,
        },
        'cluster_sizes': cluster_sizes
    }

    return S, stats, additional_data
