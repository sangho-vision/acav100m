import math
# import copy
import random
from collections import defaultdict

from tqdm import tqdm

from .common import (
    categorize_data_view, get_shuffled_ids,
    wrapped_run_derangements, shuffle_each_view
)


def get_keys(views, deranged_classes_ratio, shuffle_each_cluster=False,
             if_shuffle_classes=True):
    # in case of mnist-fdss, the sorting keeps pairing between same digits
    # clip classes to the same number
    keys = {view: sorted(list(classes.keys())) for view, classes in views.items()}
    len_keys = [len(key) for key in keys.values()]
    min_len_keys = min(len_keys)
    if max(len_keys) != min_len_keys:
        tqdm.write("class num difference: clipping to {} classes".format(min_len_keys))
        keys = {view: key[:min_len_keys] for view, key in keys.items()}
    nclasses = min_len_keys

    num_deranged_classes = math.floor(deranged_classes_ratio * nclasses)
    num_matched_classes = nclasses - num_deranged_classes
    tqdm.write("shuffling {}/{} classes".format(num_deranged_classes, nclasses))
    # random shuffle classes
    if if_shuffle_classes:
        if shuffle_each_cluster:
            for key in keys.keys():
                random.shuffle(keys[key])
        else:
            ids = list(range(len(list(keys.values())[0])))
            random.shuffle(ids)
            for key in keys.keys():
                keys[key] = [keys[key][idx] for idx in ids]

    return keys, nclasses, num_matched_classes


def cut_class_datapoints(views, class_datapoints_threshold,
                         keys, nclasses, num_matched_classes,
                         shuffle_datapoints=True,
                         if_shuffle_each_view=True,
                         fix_class_sizes=False,
                         return_class_dict=False):
    if return_class_dict:
        all_features = defaultdict(dict)
    else:
        all_features = defaultdict(list)
    subset_size = 0
    dataset_size = 0
    clipped = False
    if fix_class_sizes:
        min_size = min(min(len(v) for v in view.values()) for view in views.values())
        if class_datapoints_threshold is not None:
            min_size = min(min_size, class_datapoints_threshold)

    for i in range(nclasses):
        class_keys = {view: key[i] for view, key in keys.items()}
        view_classes = {view: views[view][key] for view, key in class_keys.items()}
        # clip datapoints to the same number
        len_datapoints = [len(data) for data in view_classes.values()]
        min_len_datapoints = min(len_datapoints)
        max_len_datapoints = max(len_datapoints)

        if class_datapoints_threshold is not None:
            min_len_datapoints = min(class_datapoints_threshold, min_len_datapoints)

        if fix_class_sizes:
            min_len_datapoints = min_size

        if max_len_datapoints != min_len_datapoints:
            clipped = True
            '''
            tqdm.write("datapoint num difference: clipping to {} -> {} datapoints".format(
                max_len_datapoints, min_len_datapoints))
            '''
            view_classes = {view: data[:min_len_datapoints] for view, data in view_classes.items()}
        num_class_datapoints = min_len_datapoints
        if if_shuffle_each_view:
            view_classes = shuffle_each_view(view_classes, num_class_datapoints, shuffle_datapoints)
        for view in view_classes.keys():
            if return_class_dict:
                all_features[view][i] = view_classes[view]
            else:
                all_features[view] += view_classes[view]

        if i < num_matched_classes:
            # matched
            subset_size += num_class_datapoints
        dataset_size += num_class_datapoints

    if clipped:
        tqdm.write("clipped datapoints to match class sizes of all views")

    return all_features, keys, dataset_size, subset_size, num_matched_classes, nclasses


def get_derangements(views, deranged_classes_ratio=0.5, shuffle_true_ids=True,
                     class_datapoints_threshold=None,
                     shuffle_datapoints=True,
                     shuffle_each_cluster=False):
    all_features, keys, dataset_size, subset_size, num_matched_classes, nclasses = \
        match_classes_with_shuffle(views,
                       deranged_classes_ratio,
                       class_datapoints_threshold,
                       shuffle_datapoints,
                       shuffle_each_cluster)
    true_shuffle_ids, wrong_shuffle_ids, true_ids = \
        get_shuffled_ids(views, dataset_size, subset_size, shuffle_true_ids)
    return run_derangements(all_features, keys, dataset_size, subset_size,
                            num_matched_classes, nclasses,
                            true_shuffle_ids, wrong_shuffle_ids, true_ids)


def match_classes_with_shuffle(views,
                               deranged_classes_ratio,
                               class_datapoints_threshold,
                               shuffle_datapoints,
                               shuffle_each_cluster,
                               return_class_dict=False,
                               add_vid=False,
                               align=False,
                               if_shuffle_each_view=True,
                               if_shuffle_classes=True):
    # views = {k: categorize_data(v, align=align) for k, v in views.items()}
    views = categorize_data_view(views, add_vid=add_vid, align=align)
    keys, nclasses, num_matched_classes = get_keys(views, deranged_classes_ratio,
                                                   shuffle_each_cluster,
                                                   if_shuffle_classes)
    all_features, keys, dataset_size, subset_size, num_matched_classes, nclasses = \
        cut_class_datapoints(views, class_datapoints_threshold,
                             keys, nclasses, num_matched_classes, shuffle_datapoints,
                             if_shuffle_each_view=if_shuffle_each_view,
                             return_class_dict=return_class_dict)
    return all_features, keys, dataset_size, subset_size, num_matched_classes, nclasses


def run_derangements(*args,  **kwargs):
    res = wrapped_run_derangements(*args, **kwargs)
    return {'unsharded': res}
