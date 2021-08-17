import random
from itertools import chain
from pathlib import Path

from prepare import prepare
from common import save_output, get_stats
from run import _run
from clustering import run_clusterings, shard_clustering


def run_sharded(args, shards, nexpr=0):
    S_sharded = []
    sharded_dataset_size = shards['sharded'][0][2]
    expr_name = 'expr_{}_shard_separate_clustering_'.format(nexpr)
    for i, shard in enumerate(shards['sharded']):
        S = _run(args, *shard, expr_name='',
                 if_save=False, verbose=False)
        S = [j + sharded_dataset_size * i for j in S]
        S_sharded = [*S_sharded, *S]
    return S_sharded, expr_name


def run_sharded_shared_clustering(args, shards, nexpr=0):
    S_sharded = []
    sharded_dataset_size = shards['sharded'][0][2]
    clusterings = get_shared_clusterings(args,
                                         shards['unsharded'], shards['sharded'],
                                         shards['sharded_ids'])
    expr_name = 'expr_{}_shard_shared_clustering_'.format(nexpr)
    for i, shard in enumerate(shards['sharded']):
        S = _run(args, *shard, expr_name='',
                 clustering=clusterings[i], if_save=False, verbose=False)
        S = [j + sharded_dataset_size * i for j in S]
        S_sharded = [*S_sharded, *S]
    return S_sharded, expr_name


def run_unsharded(args, train, shards, nexpr=0):
    expr_name = 'expr_{}'.format(nexpr)
    S_unsharded = _run(args, train_data=train, expr_name='',
                       *shards['unsharded'], if_save=False, verbose=False)
    return S_unsharded, expr_name


def get_shared_clusterings(args, full, shards, sharded_ids):
    features = full[0]
    clustering = run_clusterings(args, features, args.ncentroids, args.kmeans_iters,
                                 args.clustering_func_type)
    keys = list(clustering.keys())
    views_clusterings = [{} for _ in keys]
    for key in keys:
        # ids = [shard[1][key] for shard in sharded_ids]
        clusterings = shard_clustering(clustering[key], sharded_ids)
        for i, sharded_clustering in enumerate(clusterings):
            views_clusterings[i][key] = sharded_clustering
    return views_clusterings


def merge_shards(shards, shuffle=False):
    # shard_map = [{p:i for p in shard} for i, shard in enumerate(shards)]
    # shard_map = dict(ChainMap(*shard_map))
    merged = list(chain(*shards))
    if shuffle:
        random.shuffle(merged)
    return merged


def run_exp(verbose=True, **kwargs):
    nexprs = kwargs.get('nexprs', 10)
    kwargs['nexprs'] = 1
    args, shards = prepare(verbose=verbose, **kwargs)
    # only support training for unsharded dataset
    _, shards = shards
    true_ids = shards['unsharded'][1]
    if args['num_shards'] is None:
        print('run_unsharded')
    else:
        if args['share_clustering']:
            print('run_sharded_shared_clustering')
        else:
            print('run_sharded')
    for i in range(nexprs):
        kwargs['seed'] = args.seed + i
        if args['num_shards'] is None:
            args, shards = prepare(verbose=verbose, **kwargs)
            train, shards = shards
            S, expr_name = run_unsharded(args, train, shards, nexpr=i)
        else:
            args, shards = prepare(verbose=verbose, **kwargs)
            train, shards = shards
            if args['share_clustering']:
                S, expr_name = run_sharded_shared_clustering(args, shards, nexpr=i)
            else:
                S, expr_name = run_sharded(args, shards, nexpr=i)
        true_ids = shards['unsharded'][1]
        stats = get_stats_iter(true_ids, S)
        if args.save:
            save_output(Path(args.out_path), stats, {}, i, expr_name, args)


def get_stats_iter(gold_standard, S):
    stats = []
    for i in range(1, len(S) + 1):
        stats.append(get_stats(gold_standard, S[:i]))
    stats = list(zip(*stats))
    stats = dict(zip(['precision', 'recall', 'f1'], stats))
    return stats
