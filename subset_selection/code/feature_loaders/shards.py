import copy
from pathlib import PosixPath

from braceexpand import braceexpand

from mps import distributed as du
from .meta import load_metadata


def get_shards_size(args, shards_path):
    '''
    metadata, shards_size, shard_size_dt = load_metadata(args, shards_path)
    shards_size = shards_size[:len(shards_path)]
    return {'metadata': metadata, 'shards_size': shards_size,
            'shards_size_dt': shard_size_dt}
            '''
    metadata, shard_size_dt = load_metadata(args, shards_path)
    return {'shards_size_dt': shard_size_dt, 'metadata': metadata}


def get_shards_path(args, shards_path, suffix='.tar', f=lambda x: {}):
    '''
    path = args.data.media.path  # e.g. .../shard-{000000..000019}.{suffix}
    shard_path, shard_name = path.parent, path.stem
    shards_path = str(shard_path / "{}{}".format(shard_name, suffix))
    '''
    if isinstance(shards_path, PosixPath):
        shards_path = str(shards_path)
    if isinstance(shards_path, str):
        shards_path = list(braceexpand(shards_path))
    shards_path = sorted(shards_path)

    if args.computation.discard_shards:
        _num_shards = (
            args.computation.num_gpus *
            (len(shards_path) // args.computation.num_gpus)
        )
        if _num_shards != len(shards_path):
            p_str = f"num_shards {len(shards_path)} "
            p_str += f"do not divided by num_gpus {args.computation.num_gpus}.. "
            p_str += f"dropping last {len(shards_path) - _num_shards} shards {shards_path[_num_shards:]}"
            print(p_str)
        shards_path = shards_path[:_num_shards]
    # rank = args.process_rank if hasattr(args, 'process_rank') else None
    # total = args.computation.num_gpus
    # shards_path = du.node_selection(shards_path, rank, total)
    all_shards_path = copy.deepcopy(shards_path)  # store global shards path list
    rest = f(args, shards_path)  # run for global shards path list
    shards_path = du.node_selection(shards_path, total=du.get_world_size(), is_train=True,
                                    no_str_ok=True)
    rest['all_shards_path'] = all_shards_path
    return shards_path, rest
