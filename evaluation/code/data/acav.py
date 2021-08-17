import os
import warnings
from pathlib import Path
from functools import partial

from torch.utils.data import IterableDataset
import webdataset as wds

import utils.logging as logging

from data.contrast import VideoDecoder
from data.build import DATASET_REGISTRY
from utils import distributed as du


logger = logging.get_logger(__name__)


class SplitByNode:
    """Selects a subset of urls based on Torch get_rank/get_world_size.

    Used as a shard selection function in Dataset."""

    def __init__(self, group=None):
        self.rank = -1
        self.size = -1
        try:
            import torch
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                return
        except Exception as e:
            print(e)
            return
        if group is None:
            # group = torch.distributed.group.WORLD
            try:
                # some versions of torch don't like group=None
                import torch.distributed.distributed_c10d
                group = torch.distributed.distributed_c10d._default_pg
            except:
                pass
        self.rank = torch.distributed.get_rank(group=group)
        self.size = torch.distributed.get_world_size(group=group)

    def __call__(self, urls):
        urls = [url for url in urls]
        assert isinstance(urls, list)
        if self.size > 1:
            if self.rank == 0 and len(urls) < self.size:
                warnings.warn(f"world_size {self.size} > num_shards {len(urls)}")
            return urls[self.rank::self.size]
        else:
            return urls


def split_by_worker(urls):
    """Selects a subset of urls based on Torch get_worker_info.

    Used as a shard selection function in Dataset."""
    import torch

    urls = [url for url in urls]

    assert isinstance(urls, list)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        wid = worker_info.id
        num_workers = worker_info.num_workers
        if wid == 0 and len(urls) < num_workers:
            warnings.warn(f"num_workers {num_workers} > num_shards {len(urls)}")
        return urls[wid::num_workers]
    else:
        return urls


def shard_selection(urls, nodesplitter, splitter):
    return splitter(nodesplitter(urls))



@DATASET_REGISTRY.register()
def ACAV(cfg, mode):
    """
    ACAV video loader with VideoDecoder. Videos are stored in POSIX tar
    archives and we process them using WebDataset.
    Args:
        cfg (CfgNode): configs.
        mode (string): Options include `pretrain` mode.
    """
    assert mode in [
        "pretrain"
    ], "Split '{}' not supported for ACAV".format(mode)

    shards_path = sorted(Path(cfg.DATASET_DIR).glob("*.tar"))
    assert cfg.DATA_LOADER.NUM_WORKERS <= len(shards_path)
    s_idx = int(shards_path[0].stem.split("-")[1])
    e_idx = int(shards_path[-1].stem.split("-")[1])

    url = os.path.join(
        f"{cfg.DATASET_DIR}",
        f"shard-{{{s_idx:06d}..{e_idx:06d}}}.tar"
    )

    videodecoder = VideoDecoder(cfg)
    batch_size = int(cfg.PRETRAIN.BATCH_SIZE / du.get_world_size())
    if cfg.DATA_LOADER.NUM_WORKERS > 0:
        length = int(cfg.PRETRAIN.DATASET_SIZE / (cfg.DATA_LOADER.NUM_WORKERS * du.get_world_size()))
        nominal = int(length / batch_size) * cfg.DATA_LOADER.NUM_WORKERS * batch_size
    else:
        nominal = int(cfg.PRETRAIN.DATASET_SIZE / du.get_world_size())
        length = nominal
    nodesplitter = SplitByNode()
    _shard_selection = partial(shard_selection, nodesplitter=nodesplitter, splitter=split_by_worker)
    dataset = wds.Dataset(url, handler=wds.warn_and_continue, shard_selection=_shard_selection).shuffle(1000).map(videodecoder.decode, handler=wds.warn_and_continue)
    dataset = wds.ResizedDataset(dataset, length, nominal)

    return dataset
