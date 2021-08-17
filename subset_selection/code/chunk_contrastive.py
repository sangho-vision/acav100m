import os
import math
import copy
from functools import reduce
from multiprocessing import Pool
from pathlib import Path

from braceexpand import braceexpand
import torch.multiprocessing as mp

from save import save_output, merge_csvs, group_cache_paths
from utils import get_chunks, split_chunks, dump_pickle, load_pickle
from mps.multiprocessing import spawn
from run_contrastive import _run


def run_chunks_contrastive(args):
    args.parent_pid = str(os.getpid())
    shards_path = args.data.path
    shard_paths = sorted(list(braceexpand(str(shards_path))))
    shard_paths = [p for p in shard_paths if Path(p).is_file()]
    chunks = list(get_chunks(shard_paths, args.chunk_size))
    num_chunks = len(chunks)
    mp.set_start_method("spawn")

    if args.computation.num_gpus > num_chunks:
        print("num_gpus ({}) exceeds num_chunks ({})".format(args.computation.num_gpus,
                                                             num_chunks))
        print("thresholding num_gpus to be equal to num_chunks")
        args.computation.num_gpus = num_chunks

    '''
    res = mp.Queue()
    metas = mp.Queue()
    '''
    chunks = list(enumerate(chunks))
    nodes_chunks = list(split_chunks(chunks, args.computation.num_gpus))

    chunk_args = copy.deepcopy(args)

    if hasattr(chunk_args.subset, 'size') and isinstance(chunk_args.subset.size, int):
        chunk_args.subset.size = math.ceil(chunk_args.subset.size / num_chunks)
    chunk_args.computation.num_workers = round(chunk_args.computation.num_workers /
                                               chunk_args.computation.num_gpus)

    print("running {} chunks in {} gpus".format(num_chunks,
                                                chunk_args.computation.num_gpus))
    nodes_args = (chunk_args, nodes_chunks, num_chunks)
    spawn(run_chunks_node, chunk_args, nodes_args)


def reduce_all_pkls(args):
    cache_dir = args.data.output.path.parent / 'caches'
    paths = list(cache_dir.glob('cache_*_*.pkl'))
    groups = group_cache_paths(paths)

    for key in sorted(list(groups.keys())):
        print('processing cache set {}'.format(key))
        paths = groups[key]
        pool_args = [(args, p.stem, p) for p in paths]
        with Pool(args.computation.num_workers) as p:
            res = p.map(reduce_single_cache, pool_args)

        out_paths, counts = zip(*res)
        counts = sum(counts)
        out_paths = dict(zip(paths, out_paths))
        for k, v in out_paths.items():
            if v is None:
                print("cache not processed: {}".format(k))

        if len(out_paths) == 0:
            print("No files saved")

        print("merging csvs")
        out_path = args.data.output.path
        counts = merge_csvs(sorted(list(out_paths.values())), out_path)

        if args.verbose:
            print("Saved Results: added {} lines to {}".format(counts, out_path))


def reduce_single_cache(pool_args):
    args, k, path = pool_args
    print("loading cache ({})".format(k))
    cache = load_pickle(path)
    res = cache['res']
    metas = cache['metas']
    return _reduce_single_cache(args, k, res, metas)


def _reduce_single_cache(args, k, res, metas):
    # already merged
    # metas = reduce(lambda x, y: {**x, **y}, metas.values())  # merge dict

    if hasattr(args.subset, 'size') and isinstance(args.subset.size, int):
        res = res[:args.subset_size]

    print("saving cache ({}), subset size: {}".format(k, len(res)))
    results = [res]

    name = args.data.output.path.name
    possible_out_path = args.data.output.path.parent / 'caches' / name
    out_path = None
    counts = 0
    for samples_list in results:
        out_path, count = save_output(samples_list, metas, possible_out_path, k + '_')
        counts += count

    return out_path, count


def run_chunks_node(node_rank, cfg):
    args, chunks, num_chunks = cfg
    args.node_rank = node_rank
    chunks = chunks[node_rank]
    for i, chunk in enumerate(chunks):
        num, chunk = chunk
        print("running chunk {}".format(num))
        chunk_args = copy.deepcopy(args)
        chunk_args.chunk_num = i
        res, metas = run_chunk(chunk_args, chunk)
        pid = args.parent_pid
        name = "contrastive_cache_{}_{}_{}".format(pid, node_rank, i)
        if args.save_cache_as_csvs:
            _reduce_single_cache(args, name, res, metas)
        else:
            save_chunk_cache(args, node_rank, i, res, metas)

        # print("{}/{} chunks done".format(comp, num_chunks))
        # print("ran chunk {}".format(num))


def save_chunk_cache(args, rank, i, res, metas):
    cache_dir = args.data.output.path.parent / 'caches'
    cache_dir.mkdir(parents=True, exist_ok=True)
    pid = args.parent_pid
    dt = {
        'res': res,
        'metas': metas
    }
    name = "contrastive_cache_{}_{}_{}.pkl".format(pid, rank, i)
    path = str(cache_dir / name)
    dump_pickle(dt, path)


def run_chunk(args, chunk):
    results, metas = _run(args, chunk)
    return results[0], metas  # assume single partition
