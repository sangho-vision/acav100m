import os
import math
import copy
from pathlib import Path
from collections import OrderedDict

import torch
from braceexpand import braceexpand

from feature_loaders import get_dataloader
from measures.contrastive import Contrastive
from save import save_output
import mps.multiprocessing as mpu
import mps.distributed as du


def _run(args, path):
    measure = Contrastive(args.contrastive.num_epochs, 'cpu',
                          args.contrastive.base_lr,
                          args.contrastive.num_warmup_steps)
    if args.contrastive.cached_epoch is not None:
        cache_path = measure.get_cache_path_load(args, path, args.contrastive.cached_epoch)
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.is_file():
                if args.verbose:
                    print("cache file found: {}".format(cache_path.stem))
                    print("loading from cached file")
                measure.load_cache(args, path, args.contrastive.cached_epoch)
                if args.contrastive.train_from_cached:
                    if args.verbose:
                        print("training from cached file")
                    measure = train(args, path=path, measure=measure)
            else:
                if args.verbose:
                    print("cache file not found: {}".format(cache_path.stem))
                    print("training from scratch")
                measure = train(args, path)
        else:
            if args.verbose:
                print("no cache file found")
                print("training from scratch")
            measure = train(args, path)
    else:
        measure = train(args, path)

    infer(args, path, measure)
    '''
    samples_list, metas = infer(args, path, measure)
    results = []
    results.append(samples_list)
    return results, metas
    '''


def train(args, path, measure=None):
    if args.computation.use_distributed:
        return train_distributed(args, path=path, measure=measure)
    else:
        return _train(args, path=path, measure=measure)


def copy_measure(args, measure, prev_measure):
    measure.model.visual_linear.weight.data = copy.deepcopy(prev_measure.model.visual_linear.weight.data.to(args.computation.device))
    measure.model.audio_linear.weight.data = copy.deepcopy(prev_measure.model.audio_linear.weight.data.to(args.computation.device))
    measure.model.visual_linear.requires_grad_(True)
    measure.model.audio_linear.requires_grad_(True)
    measure.epoch = prev_measure.epoch
    return measure


def _train(args, path, measure=None, distributed=False):
    prev_measure = measure
    measure = Contrastive(args.contrastive.num_epochs, args.computation.device,
                          args.contrastive.base_lr,
                          args.contrastive.num_warmup_steps,
                          distributed)
    if prev_measure is not None:
        measure = copy_measure(args, measure, prev_measure)

    print("training contrastive loss")
    dataloader, _ = get_dataloader(args, path, args.contrastive.train_batch_size,
                                   shuffle=True, is_train=True)
    measure.train(args, path, dataloader, args.log_every, args.verbose)
    return measure


def infer(args, path, measure):
    if args.computation.use_distributed:
        return infer_distributed(args, measure, path)
    else:
        scores, res, metas = _infer(args, path, measure)
        return res, metas


def _infer(args, path, measure, distributed=False):
    prev_measure = measure
    measure = Contrastive(args.contrastive.num_epochs, args.computation.device,
                          args.contrastive.base_lr,
                          args.contrastive.num_warmup_steps,
                          distributed)
    measure = copy_measure(args, measure, prev_measure)

    print("(node {}) running inference".format(du.get_rank()))
    dataloader, metas = get_dataloader(args, path, args.contrastive.test_batch_size,
                                   shuffle=True, is_train=False)  # is_train means duplicating dataloader for all nodes
    dataset_size = dataloader.dataset.dataset_size
    subset_size = args.subset.size
    if subset_size is None:
        subset_size = math.inf
    scores, ids, meta_rows = measure.infer(args, dataloader, metas, subset_size, args.log_every, args.verbose)
    ids = ids.numpy().tolist()
    res = [meta_rows[idx] for idx in ids]
    print("(node {}) done inference".format(du.get_rank()))
    return scores, res, metas


def train_distributed(args, path, measure=None):
    print("training contrastive loss with distributed")
    du.set_environment_variables_for_nccl_backend(
        du.get_global_size() == du.get_local_size(args.computation.num_gpus),
        args.computation.master_port,
        # args.computation.use_distributed,
        False
    )
    if measure is None:
        measure = Contrastive(args.contrastive.num_epochs, 'cpu',
                            args.contrastive.base_lr,
                            args.contrastive.num_warmup_steps,
                            distributed=True)
    # args.min_shard_num = math.floor(len(path) / args.computation.num_gpus)
    cfg = (args, measure, path)

    torch.multiprocessing.spawn(
        mpu.run_local,
        nprocs=args.computation.num_gpus,
        args=(
            args.computation.num_gpus,
            _train_distributed,
            args.computation.dist_init_method,
            args.computation.shard_id,
            args.computation.num_shards,
            args.computation.dist_backend,
            cfg,
        ),
        daemon=False,
    )
    cache_dir = args.data.output.path.parent / 'caches'
    pid = args.parent_pid
    name = "contrastive_trained_model_cache_{}.pkl".format(pid)
    cache_path = cache_dir / name
    dt = torch.load(cache_path)
    measure.base_lr = dt['base_lr']
    measure.model.load_state_dict(dt['model'])

    return measure


def _train_distributed(local_rank, cfg):
    args, measure, path = cfg
    measure = _train(args, measure=measure, path=path, distributed=True)
    if du.is_master_proc():
        cache_dir = args.data.output.path.parent / 'caches'
        cache_dir.mkdir(parents=True, exist_ok=True)
        pid = args.parent_pid
        name = "contrastive_trained_model_cache_{}.pkl".format(pid)
        path = cache_dir / name
        dt = {
            'base_lr': measure.base_lr,
            'model': measure.model.state_dict()
        }
        torch.save(dt, path)
    else:
        return None


def infer_distributed(args, measure, path):
    print("inferring with distributed")
    du.set_environment_variables_for_nccl_backend(
        du.get_global_size() == du.get_local_size(args.computation.num_gpus),
        args.computation.master_port,
        False
    )
    cfg = (args, measure, path)
    torch.multiprocessing.spawn(
        mpu.run_local,
        nprocs=args.computation.num_gpus,
        args=(
            args.computation.num_gpus,
            _infer_distributed,
            args.computation.dist_init_method,
            args.computation.shard_id,
            args.computation.num_shards,
            args.computation.dist_backend,
            cfg,
        ),
        daemon=False,
    )
    '''
    print("loading caches from workers")
    cache_dir = args.data.output.path.parent / 'caches'
    pid = args.parent_pid
    name = "contrastive_inferred_cache_{}_*.pkl".format(pid)
    data = []
    metas = {}
    for path in cache_dir.glob(name):
        dt = torch.load(path)
        data.extend(dt['data'])
        # metas = {**metas, **dt['metas']}
        metas = dt['metas']
        print(name, len(dt['data']))

    print("sorting outputs")
    data = sorted(data, key=lambda x: x[0], reverse=True)
    print("formatting outputs")
    data = [(row[1]['filename'], row[1]) for row in data]
    # remove duplicates
    print("removing duplicates")
    data = list(OrderedDict(data).values())
    return data, metas
    '''


def _infer_distributed(local_rank, cfg):
    args, measure, path = cfg
    scores, res, metas = _infer(args, measure=measure, path=path, distributed=True)
    data = list(zip(scores, res))

    '''
    cache_dir = args.data.output.path.parent / 'caches'
    cache_dir.mkdir(parents=True, exist_ok=True)
    pid = args.parent_pid
    name = "contrastive_inferred_cache_{}_{}.pkl".format(pid, local_rank)
    path = cache_dir / name
    dt = {
        'data': data,
        'metas': metas
    }
    print("(node {}) saving cache on completion".format(du.get_rank()))
    torch.save(dt, path)
    print("(node {}) saved cache on completion".format(du.get_rank()))
    '''


def run_single_contrastive(args):
    args.parent_pid = str(os.getpid())
    shards_path = args.data.path
    shard_paths = sorted(list(braceexpand(str(shards_path))))
    shard_paths = [p for p in shard_paths if Path(p).is_file()]
    # extract full
    args.subset.size = None
    args.subset.ratio = 1.0
    _run(args, shard_paths)
    print("done")
    '''
    results, metas = _run(args, shard_paths)
    counts = 0

    print("sorted all data by similarity score: dataset_size {}".format(len(results[0])))

    out_path = None
    for samples_list in results:
        print("saving outputs to {}".format(args.data.output.path))
        out_path, count = save_output(samples_list, metas, args.data.output.path,
                                      sharded_meta=True)
        counts += count

    if out_path is None:
        print("No files saved")

    if args.verbose:
        print("Saved Results: added {} lines to {}".format(counts, out_path))
    '''
