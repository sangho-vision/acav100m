import os
from pathlib import Path
import math

from tqdm import tqdm
from braceexpand import braceexpand

import torch

from run_extraction import run_extraction
from models import download_model_weights

import mps.multiprocessing as mpu
import mps.distributed as du
from utils import to_brace
from save import get_processed_paths


def parallel_extraction_script(args):
    path = Path(args.data.path)
    paths = [path.parent / shard for shard in braceexpand(path.name)]
    processed = paths
    args.computation.num_gpus = min(args.computation.num_gpus, len(paths))
    if len(paths) == 0:
        print(f"All shards of {args.data.path} processing already done!")
    else:
        suffix = paths[0].suffix
        path = paths[0].parent / "{}{}".format(to_brace(list(set([p.stem for p in paths]))), suffix)
        num_shards = len(list(braceexpand(path.name)))

        # download model weights once before spawning processes
        download_model_weights(args)

        total_files_num = len(processed) + num_shards
        print("{}/{} files already processed".format(len(processed), total_files_num))
        print("extracting remaining {} files".format(num_shards))
        args.data.media.path = path
        args.computation.num_gpus = min(args.computation.num_gpus, num_shards)
        if args.computation.num_gpus > 1:
            du.set_environment_variables_for_nccl_backend(
                du.get_global_size() == du.get_local_size(args.computation.num_gpus),
                args.computation.master_port,
                args.computation.use_distributed,
            )
            if args.computation.use_distributed:
                mpu.run(
                    du.get_local_rank(args.computation.num_gpus),
                    run_extraction,
                    args.computation.dist_backend,
                    args,
                )
            else:
                torch.multiprocessing.spawn(
                    mpu.run_local,
                    nprocs=args.computation.num_gpus,
                    args=(
                        args.computation.num_gpus,
                        run_extraction,
                        args.computation.dist_init_method,
                        args.computation.shard_id,
                        args.computation.num_shards,
                        args.computation.dist_backend,
                        args,
                    ),
                    daemon=False,
                )
        else:
            out_paths = run_extraction(args)
            print("extracted {} files".format(len(out_paths)))
    return get_processed_paths(args, paths)
