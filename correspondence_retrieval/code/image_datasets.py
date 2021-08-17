from itertools import chain
from functools import partial

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def merge_datasets(datasets):
    if isinstance(datasets, dict):
        keys = sorted(list(datasets.keys()))
        datasets = [datasets[key] for key in keys]
    res = datasets[0]
    if torch.is_tensor(datasets[0].data):
        res.data = torch.cat([dataset.data for dataset in datasets], dim=0)
    else:
        res.data = np.concatenate([dataset.data for dataset in datasets], axis=0)
    res.targets = list(chain(*[dataset.targets for dataset in datasets]))
    return res


def get_cifar10(path, rotation=False, flip=False):
    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]  # color normalization
    transform = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    if rotation:
        Rot90 = partial(
            transforms.functional.rotate,
            angle=90,
        )
        transform = [Rot90, *transform]

    if flip:
        VFlip = transforms.functional.vflip
        transform = [VFlip, *transform]

    transform = transforms.Compose(transform)

    datasets = {
        'train': torchvision.datasets.CIFAR10(
            root=str(path), train=True, download=True, transform=transform),
        'test': torchvision.datasets.CIFAR10(
            root=str(path), train=False, download=True, transform=transform)
    }
    datasets = merge_datasets(datasets)
    datasets.name = 'CIFAR10'
    return datasets


def get_cifar10_rotated(path):
    return get_cifar10(path, rotation=True)


def get_cifar10_flipped(path):
    return get_cifar10(path, flip=True)


def get_mnist(path):
    # MNIST categories are imbalanced
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    datasets = {
        'train': torchvision.datasets.MNIST(root=str(path), train=True, transform=transform, download=True),
        'test': torchvision.datasets.MNIST(root=str(path), train=False, transform=transform, download=True)
    }
    dataset = merge_datasets(datasets)
    dataset.name = 'MNIST'
    return dataset


dataset_dict = {
    'cifar10': get_cifar10,
    'cifar10-rotated': get_cifar10_rotated,
    'cifar10-flipped': get_cifar10_flipped,
    'mnist': get_mnist
}
