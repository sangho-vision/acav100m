import math
import random
import copy
from itertools import chain

import torch

from utils import peek


def get_penultimates(keys):
    penultimates = {}
    for key in keys:
        view = key[:key.find('_')]  # get dataset+model name
        layer_name = key[key.find('_') + 1:]
        if view not in penultimates:
            penultimates[view] = view + '_' + layer_name
        elif layer_name > penultimates[view]:
            penultimates[view] = view + '_' + layer_name
    keys = sorted(list(penultimates.keys()))
    return [penultimates[k] for k in keys]


class ClassDataLoader:
    def __init__(self, data):
        dataset, labels, num_iters = self.shuffle_dataset(data)
        self.dataset = dataset
        self.labels = labels
        self.num_iters = num_iters

    def shuffle_dataset(self, data):
        dataset = {}
        lengths = {}
        for label, class_data in data.items():
            class_data = copy.deepcopy(class_data)
            random.shuffle(class_data)
            lengths[label] = len(class_data)
            dataset[label] = class_data

        max_len = max(lengths.values())
        self.batch_size = self.num_classes = len(data)
        # num_iters = math.ceil(max_len / batch_size)
        num_iters = max_len
        labels = sorted(list(data.keys()))

        return dataset, labels, num_iters

    def gather_batch(self, i):
        batch = [self.get_index(self.dataset[label], i) for label in self.labels]
        return batch

    def __iter__(self):
        for i in range(self.num_iters):
            batch = self.gather_batch(i)
            batch = [self.format_row(row) for row in batch]
            batch = zip(*batch)  # zip by feature type
            batch = [torch.stack(v, dim=0) for v in batch]
            yield batch

    def __len__(self):
        return self.num_iters

    def get_index(self, cset, i):
        remainder = i % len(cset)
        return cset[remainder]

    def format_row(self, row):
        penultimates = get_penultimates(list(row['features'].keys()))
        return [row['features'][k] for k in penultimates]


class SampleDataLoader(ClassDataLoader):
    def __init__(self, data, batch_size):
        dataset = self.shuffle_dataset(data)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iters = math.ceil(len(self.dataset) / self.batch_size)

    def shuffle_dataset(self, data):
        data = list(chain(*data.values()))  # ignore classes
        random.shuffle(data)
        return data

    def gather_batch(self, i):
        start = self.batch_size * i
        end = self.batch_size * (i + 1)
        batch = self.dataset[start: end]
        return batch


class InferDataLoader(SampleDataLoader):
    def __init__(self, data, batch_size):
        self.dataset = self.get_dataset(data)
        self.batch_size = batch_size
        self.num_iters = math.ceil(len(self.dataset) / self.batch_size)

    def get_dataset(self, data):
        penultimates = get_penultimates(list(data.keys()))
        dataset = []
        for i in range(len(peek(data))):
            features = {k: data[k][i] for k in penultimates}
            row = {'features': features}
            dataset.append(row)
        return dataset
