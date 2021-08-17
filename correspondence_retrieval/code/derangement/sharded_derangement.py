import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from .common import (
    categorize_data, wrapped_run_derangements,
    shuffle_each_view, get_shuffled_ids
)
from .derangement import get_keys, cut_class_datapoints


def sharded_cut_class_datapoints(views, class_datapoints_threshold,
                                 keys, nclasses, num_matched_classes,
                                 shuffle_datapoints=True,
                                 num_shards=10):
    all_features = [defaultdict(list) for _ in range(num_shards)]
    subset_size = 0
    dataset_size = 0
    clipped = False
    for i in range(nclasses):
        class_keys = {view: key[i] for view, key in keys.items()}
        view_classes = {view: views[view][key] for view, key in class_keys.items()}
        # clip datapoints to the same number
        len_datapoints = [len(data) for data in view_classes.values()]
        min_len_datapoints = min(len_datapoints)
        max_len_datapoints = max(len_datapoints)

        if class_datapoints_threshold is not None:
            min_len_datapoints = min(class_datapoints_threshold, min_len_datapoints)

        shard_size = min_len_datapoints // num_shards
        min_len_datapoints = min(shard_size * num_shards, min_len_datapoints)

        if max_len_datapoints != min_len_datapoints:
            clipped = True
            '''
            tqdm.write("datapoint num difference: clipping to {} -> {} datapoints".format(
                max_len_datapoints, min_len_datapoints))
            '''
            view_classes = {view: data[:min_len_datapoints] for view, data in view_classes.items()}
        num_class_datapoints = min_len_datapoints
        view_classes = shuffle_each_view(view_classes, num_class_datapoints, shuffle_datapoints)
        for view in view_classes.keys():
            shards = shard_datapoints(view_classes[view], num_shards)
            for j, shard in enumerate(shards):
                all_features[j][view] += shard

        if i < num_matched_classes:
            # matched
            subset_size += shard_size
        dataset_size += shard_size

    if clipped:
        tqdm.write("clipped datapoints to match class sizes of all views")

    args = [(features, keys, dataset_size, subset_size, num_matched_classes, nclasses)
            for features in all_features]
    return args


def shard_datapoints(x, num_shards=10):
    shard_size = len(x) // num_shards
    x = x[:shard_size * num_shards]
    return shard_with_size(x, shard_size)


def shard_with_size(x, shard_size):
    return [x[i:i + shard_size] for i in range(0, len(x), shard_size)]


def merge(x, size):
    x = np.array(x)  # num_shard, shard_size
    num = x.shape[0]
    stair = np.arange(num)[:, None] * size
    x += stair
    x = x.reshape(-1)
    x = x.tolist()
    random.shuffle(x)
    return x


def get_shuffled_ids_with_shards(views, dataset_size, subset_size,
                                 sharded_dataset_size, sharded_subset_size, shuffle_true_ids=True):
    # randomness involved
    num_shards = dataset_size // sharded_dataset_size

    def get_with_shards(size):
        sharded = [list(range(size)) for _ in range(num_shards)]
        for x in sharded:
            random.shuffle(x)
        unsharded = merge(sharded, size)
        return unsharded, sharded

    true_shuffle_ids, true_shuffle_ids_sharded = get_with_shards(sharded_subset_size)

    if shuffle_true_ids:
        print("shuffled true_ids")
        true_ids_sharded = [sorted(random.sample(list(range(sharded_dataset_size)), sharded_subset_size))
                            for _ in range(num_shards)]
        true_ids = merge(true_ids_sharded, sharded_dataset_size)
    else:
        true_ids, true_ids_sharded = get_with_shards(sharded_subset_size)

    wrong_shuffle_ids = {}
    wrong_shuffle_ids_sharded = [{} for _ in range(num_shards)]
    for view in views.keys():
        wrong_shuffle_ids[view], sharded = \
            get_with_shards(sharded_dataset_size - sharded_subset_size)
        for i, shard in enumerate(sharded):
            wrong_shuffle_ids_sharded[i][view] = shard

    return true_shuffle_ids, wrong_shuffle_ids, true_ids, \
        true_shuffle_ids_sharded, wrong_shuffle_ids_sharded, true_ids_sharded


def get_sharded_derangements(views, deranged_classes_ratio=0.5, shuffle_true_ids=True,
                             class_datapoints_threshold=None,
                             shuffle_datapoints=True,
                             shuffle_each_cluster=False,
                             num_shards=10):
    views = {k: categorize_data(v) for k, v in views.items()}
    keys, nclasses, num_matched_classes = get_keys(views, deranged_classes_ratio,
                                                   shuffle_each_cluster)
    unsharded_args = cut_class_datapoints(views, class_datapoints_threshold,
                                          keys, nclasses, num_matched_classes)
    '''
    sharded_args = sharded_cut_class_datapoints(views, class_datapoints_threshold,
                                                keys, nclasses, num_matched_classes,
                                                shuffle_datapoints,
                                                num_shards=num_shards)
                                                '''
    dataset_size = unsharded_args[2]
    subset_size = unsharded_args[3]
    '''
    sharded_dataset_size = sharded_args[0][2]
    sharded_subset_size = sharded_args[0][3]
    true_shuffle_ids, wrong_shuffle_ids, true_ids, \
        true_shuffle_ids_sharded, wrong_shuffle_ids_sharded, true_ids_sharded = \
        get_shuffled_ids_with_shards(views, dataset_size, subset_size,
                                     sharded_dataset_size, sharded_subset_size, shuffle_true_ids)
    '''
    true_shuffle_ids, wrong_shuffle_ids, true_ids = \
            get_shuffled_ids(views, dataset_size, subset_size, shuffle_true_ids)

    # true_shuffle_ids, wrong_shuffle_ids, true_ids = \
    #   get_shuffled_ids(views, dataset_size, subset_size, shuffle_true_ids)

    unsharded = wrapped_run_derangements(*unsharded_args, true_shuffle_ids, wrong_shuffle_ids, true_ids)
    all_features, true_ids, dataset_size, subset_size, nclasses, class_matches = \
        unsharded
    '''
    sharded_ids = list(zip(true_shuffle_ids_sharded, wrong_shuffle_ids_sharded, true_ids_sharded))
    sharded = [wrapped_run_derangements(*args, *sharded_idx)
               for args, sharded_idx in zip(sharded_args, sharded_ids)]
               '''
    sharded, sharded_ids = get_shards(all_features, true_ids, dataset_size,
                                      subset_size, nclasses, class_matches,
                                      num_shards)

    # UNBALANCED SHARDING
    return {'sharded': sharded, 'unsharded': unsharded, 'sharded_ids': sharded_ids}


def get_shards(all_features, true_ids, dataset_size, subset_size, nclasses, class_matches,
               num_shards):
    true_ids = np.array(true_ids)
    shard_size = dataset_size // num_shards
    rest = dataset_size % num_shards
    shard_sizes = [shard_size for _ in range(num_shards)]
    shard_sizes[-1] += rest
    seek = 0
    sharded = []
    sharded_ids = []
    for shard_size in shard_sizes:
        shard_dataset_size = shard_size
        start = seek
        end = start + shard_size
        shard_true_ids = (true_ids[(true_ids >= start) & (true_ids < end)] - start).tolist()
        shard_subset_size = len(shard_true_ids)
        shard_all_features = {k: v[start:end] for k, v in all_features.items()}
        sharded.append((shard_all_features, shard_true_ids,
                        shard_dataset_size, shard_subset_size, nclasses, class_matches))
        sharded_ids.append(list(range(start, end)))
        seek += shard_size

    return sharded, sharded_ids
