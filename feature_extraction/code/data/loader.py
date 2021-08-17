import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
import webdataset as wds

from .webdataset import get_dataset
from utils import identity
from mps import distributed as du


def get_dataloader(args, model, drop_last=False, shuffle=False):
    dataset, num_workers = get_dataset(args, model, shuffle=shuffle)
    media_path = args.data.media.path
    use_webdataset = (
        media_path.stem not in ['HMDB51', 'UCF101', 'FSDD', 'KineticsSounds']
    )

    if isinstance(args.computation.num_gpus, int):
        world_size = min(du.get_world_size(), args.computation.num_gpus)
    else:
        world_size = du.get_world_size()

    batch_size = int(args.data.batch_size / world_size)

    if isinstance(dataset, IterableDataset):
        shuffle = False
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=identity)
    return dataloader


def collate(batch):
    elem = batch[0]
    '''
    if isinstance(elem, tuple):
        batch = [{**elem[0], 'label': elem[1]} for elem in batch]
        elem = batch[0]
    '''
    collated = {}
    pathways_packed = False
    for key in elem.keys():
        vals = [row[key] for row in batch]
        if isinstance(elem[key], np.ndarray):
            vals = [torch.Tensor(val) for val in vals]
        # stack if possible
        if same_shape(vals):
            vals = torch.stack(vals, dim=0)
        if key == 'data' and packed_pathways(vals):
            try:
                vals = [torch.stack(v, dim=0) for v in zip(*vals)]
                pathways_packed = True
            except Exception as e:
                print(f"error stacking slowfast features within a batch: {e}")
                batch_filenames = [row['filename'] for row in batch]
                print(f"filenames in batch: {batch_filenames}")
                raise Exception
        collated[key] = vals
    options = {'pathways_packed': pathways_packed}
    return collated, options


def are_tensors(li):
    return all([torch.is_tensor(x) for x in li])


def same_shape(li):
    if not are_tensors(li):
        return False
    shapes = [x.shape for x in li]
    return all(x == shapes[0] for x in shapes)


def packed_pathways(li):
    if not isinstance(li[0], list):
        return False
    if not torch.is_tensor(li[0][0]):
        return False
    all_shapes = [[p.shape for p in pathway] for pathway in zip(*li)]
    return all([all(x == shapes[0] for x in shapes) for shapes in all_shapes])
