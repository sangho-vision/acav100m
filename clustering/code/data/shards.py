import copy
from pathlib import Path

from braceexpand import braceexpand

from .meta import load_metadata
from mps import distributed as du


def get_shards_size(args, shards_path):
    shard_size_dt = load_metadata(args, shards_path)
    return {'shards_size_dt': shard_size_dt}


def get_shards_path(args, suffix='.tar', f=lambda x: {}, is_train=False):
    path = args.data.media.path  # e.g. .../shard-{000000..000019}.{suffix}
    shard_path, shard_name = path.parent, path.stem
    shards_path = str(shard_path / "{}{}".format(shard_name, suffix))
    shards_path = sorted(list(braceexpand(shards_path)))
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
    rest = f(args, shards_path)  # run for global shards path list
    if 'shards_size_dt' in rest:
        # keep only shards with meta json files
        shards_path = [p for p in shards_path if Path(p).stem in rest['shards_size_dt']]
    all_shards_path = copy.deepcopy(shards_path)  # store global shards path list
    shards_path = du.node_selection(shards_path, total=args.computation.num_gpus, is_train=is_train)
    rest['all_shards_path'] = all_shards_path
    return shards_path, rest
