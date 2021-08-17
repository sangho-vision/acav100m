from functools import partial
from collections import defaultdict

from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms

from model import get_model


def get_loaders(num_workers):
    # Datasets and Dataloaders
    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]  # color normalization

    Rot90 = partial(
        transforms.functional.rotate,
        angle=90,
    )

    class SingleToPairTransform:
        """Pair of (original, modified)"""
        def __init__(self, funcs):
            self.funcs = funcs

        def run(self, x):
            for func in self.funcs:
                x = func(x)
            return x

        def __call__(self, x):
            return x, self.run(x)

    class PairTransform:
        def __init__(self, funcs):
            self.funcs = funcs

        def __call__(self, xs):
            res = []
            for x in xs:
                for func in self.funcs:
                    x = func(x)
                res.append(x)
            return res

    view_transform = transforms.Compose(
        [
            SingleToPairTransform([Rot90]),
            PairTransform([transforms.ToTensor()]),
            PairTransform([transforms.Normalize(mean, std)]),
        ]
    )

    datasets = {
        'train': torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=view_transform),
        'test': torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=view_transform),
    }

    loaders = {key: torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=num_workers)
        for key, dataset in datasets.items()}

    return loaders


def _get_feature(model, loaders, device):
    views_features = defaultdict(lambda: defaultdict(list))

    print("extracting features")
    # view / class / index
    with torch.no_grad():
        for loader in loaders.values():
            for views, labels in tqdm(loader, ncols=80):
                outputs = []
                for view_index, view in enumerate(views):
                    view = view.to(device)
                    outputs = model(view)
                    outputs = outputs.detach().cpu()
                    for i in range(len(labels)):
                        views_features[view_index][labels[i].item()].append(outputs[i].detach().cpu())

    dataset_size = sum(len(class_features) for class_features in views_features[0].values())
    nclasses = len(views_features[0])
    for view_index, view_features in views_features.items():
        for class_index, class_feature in view_features.items():
            views_features[view_index][class_index] = torch.stack(class_feature, dim=0)
    return views_features, dataset_size, nclasses


def get_feature(num_workers, device, finetune=False, sample=False):
    loaders = get_loaders(num_workers)
    models = get_model(device, num_workers, finetune=finetune, sample=sample)
    model = models['model']
    features, dataset_size, nclasses = _get_feature(model, loaders, device)
    return features, dataset_size, nclasses
