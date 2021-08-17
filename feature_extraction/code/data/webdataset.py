import json
import tarfile
import types
from pathlib import Path
from functools import partial

# import braceexpand
import torch
import webdataset as wds
from tqdm import tqdm

from save import load_shard_caches
from utils import get_num_workers, get_tensor_size, identity
from .shards import get_shards_size, get_shards_path
from .preprocess import Preprocessors
from .video import load_video_webdata
from .metawebdataset import MetaWebDataset
from mps import distributed as du


def _get_name(finfo, id_len=12):
    path = finfo['__key__']
    key = Path(path).stem
    filename = Path(path).name
    start = int(key[id_len:])
    return finfo['mp4'], {'idx': key, 'start': start, 'filename': filename,
                          #'shard_size': finfo['shard_size'],
                          'shard_name': finfo['shard_name']}


def _add_meta(row, shards_size_dt):
    videos, meta = row
    name = meta['shard_name']
    meta['shard_size'] = shards_size_dt[name]
    # meta = {**meta, **metadata[idx]}
    return {key: {**meta, **video} for key, video in videos.items()}


def get_dataset(args, models, shuffle=False):
    shards_path, rest = get_shards_path(args, f=get_shards_size)
    if isinstance(args.computation.num_gpus, int):
        world_size = min(du.get_world_size(), args.computation.num_gpus)
    else:
        world_size = du.get_world_size()

    batch_size = int(args.data.batch_size / world_size)
    # Here, shards_path is the list of paths to shards allocated to current node (gpu)
    # (after discarding if args.computation.discard_shards is set)
    num_workers, effective_num_workers = get_num_workers(
        args.computation.num_workers, len(shards_path)
    )

    out_str = "#Workers of Feature Extraction Dataset"
    out_str += f" (node={du.get_rank()})"
    out_str += f": {num_workers}"
    print(out_str)

    shards_size_dt = rest['shards_size_dt']
    shard_names = [Path(p).stem for p in shards_path]
    if args.acav.force_cache_restart:
        skip_lists = {}
        caches = None
        shards_size = [shards_size_dt[key] for key in shard_names]
    else:
        caches, skip_lists = load_shard_caches(args, shards_path)
        shards_size = [shards_size_dt[key] - len(skip_lists[key]) for key in shard_names]

    # print('building dataset')
    data = MetaWebDataset(shards_path, handler=wds.warn_and_continue,
                          skip_lists=skip_lists)

    id_len = 25 if args.acav.use_replicates else 12
    get_name = partial(_get_name, id_len=id_len)
    load_video = partial(load_video_webdata,
                         num_frames=args.data.media.num_frames,
                         duration=args.acav.duration,
                         skip_shorter_seconds=args.acav.duration * args.acav.skip_shorter_ratio)

    add_meta = partial(_add_meta, shards_size_dt=rest['shards_size_dt'])
    preprocess = Preprocessors(args, models)

    data = (
        data
        .map(get_name, handler=wds.warn_and_continue)
        .map_tuple(load_video, identity, handler=wds.warn_and_continue)
        .pipe(drop_none)
        .map_tuple(preprocess, identity, handler=wds.warn_and_continue)
        # .pipe(drop_none_post)
        .map(check_data_none, handler=wds.warn_and_continue)
        .map(add_meta, handler=wds.warn_and_continue)
    )

    if shuffle:
        data = data.shuffle(args.computation.shuffle_bufsize)

    '''
    if the actual number of datapoints is smaller than length,
    the ResizedDataset will fill the difference with duplicate datapoints
    '''
    # Here, shards_size is the list of sizes of all input shards, not those allocated on current
    # node (gpu)
    # (after discarding if args.computation.discard_shards is set)
    # print('resizing dataset')
    '''
    if du.get_rank() == 0:
        print("total dataset_size: {}".format(sum(shards_size)))
    '''
    print("rank {} dataset_size: {}".format(du.get_rank(), shards_size))
    length = du.get_length(
        shards_size, batch_size, args.computation.num_workers, world_size
    )
    nominal = length * effective_num_workers
    # print('X', shards_size, length, effective_num_workers, args.computation.num_workers)
    data = wds.ResizedDataset(
        data,
        length,
        nominal,
    )
    data.caches = caches

    return data, num_workers


def _decode_label(anno, class2idx):
    label = anno['annotations']['label']
    return torch.tensor(class2idx[label], dtype=torch.long)


def drop_none(it):
    for row in it:
        data, label = row
        if data is not None and \
                data[0][0] is not None and \
                data[1][0] is not None:
            yield data, label


def check_data_none(row):
    videos, meta = row
    assert meta is not None, 'metadata is None'
    assert videos is not None, 'data is None'
    for model_name, video in videos.items():
        assert isinstance(video, dict), 'data is not a dict'
        assert video['data'] is not None, 'data feature is None'
    return row


def _preprocess(data, model):
    if data is None or data[0][0] is None or data[1][0] is None:
        return None
    else:
        output = model.preprocess(*data)  # {'data': preprocessed, 'fps': fps}
        if 'data' in output and get_tensor_size(output['data']) == 0:
            return None  # no element
        else:
            return output
