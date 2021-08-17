import copy
import math
import random
from itertools import chain
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

import torch

from utils import load_pickle


def remove_labels(views):
    labels = {view: [x['label'] for x in classes] for view, classes in views.items()}
    features = {view: [x['features'] for x in classes] for view, classes in views.items()}
    return features, labels


def remove_labels_view(view):
    return [x['features'] for x in view]


def cut(dt, length):
    keys = sorted(list(dt.keys()))
    return {key: dt[key] for key in keys[:length]}


def load_chunks(path):
    chunks = list(path.glob('chunk_*.pkl'))
    data = [load_pickle(path).items() for path in chunks]
    data = dict(chain(*data))  # merge chunks
    return data


def categorize_data(data):
    classes = defaultdict(dict)
    for vid, datum in data.items():
        classes[datum['label']][vid] = datum
    for label, class_dt in classes.items():
        keys = sorted(list(class_dt.keys()))
        classes[label] = [class_dt[key] for key in keys]
    return classes


def get_derangements(views, deranged_classes_ratio=0.5, shuffle_true_ids=True,
                     class_datapoints_threshold=None):
    keys = {view: sorted(list(classes.keys())) for view, classes in views.items()}
    # clip classes to the same number
    len_keys = [len(key) for key in keys.values()]
    min_len_keys = min(len_keys)
    if max(len_keys) != min_len_keys:
        tqdm.write("class num difference: clipping to {} classes".format(min_len_keys))
        keys = {view: key[:min_len_keys] for view, key in keys.items()}
    nclasses = min_len_keys

    num_deranged_classes = math.floor(deranged_classes_ratio * nclasses)
    num_matched_classes = nclasses - num_deranged_classes
    tqdm.write("shuffling {}/{} classes".format(num_deranged_classes, nclasses))
    # random sort classes
    for key in keys.keys():
        random.shuffle(keys[key])

    all_features = defaultdict(list)
    subset_size = 0
    dataset_size = 0
    clipped = False
    for i in range(min_len_keys):
        class_keys = {view: key[i] for view, key in keys.items()}
        view_classes = {view: views[view][key] for view, key in class_keys.items()}
        # clip datapoints to the same number
        len_datapoints = [len(data) for data in view_classes.values()]
        min_len_datapoints = min(len_datapoints)
        max_len_datapoints = max(len_datapoints)

        if class_datapoints_threshold is not None:
            min_len_datapoints = class_datapoints_threshold

        if max_len_datapoints != min_len_datapoints:
            clipped = True
            '''
            tqdm.write("datapoint num difference: clipping to {} -> {} datapoints".format(
                max_len_datapoints, min_len_datapoints))
            '''
            view_classes = {view: data[:min_len_datapoints] for view, data in view_classes.items()}
        num_datapoints = min_len_datapoints
        # shuffle datapoints within each class
        for view in view_classes.keys():
            random.shuffle(view_classes[view])
            # all_features[view].append(view_classes[view])
            all_features[view] += view_classes[view]

        if i < num_matched_classes:
            # matched
            subset_size += num_datapoints
        dataset_size += num_datapoints

    if clipped:
        tqdm.write("clipped datapoints to match class sizes of all views")

    true_shuffle_ids = list(range(subset_size))
    random.shuffle(true_shuffle_ids)
    if shuffle_true_ids:
        print("shuffled true_ids")
        true_ids = sorted(random.sample(list(range(dataset_size)), subset_size))
    else:
        true_ids = list(range(subset_size))
    # true_ids = list(range(subset_size))
    for view, features in all_features.items():
        true_ids_temp = copy.deepcopy(true_ids)
        # for true matches, align all views
        true_matches = features[:subset_size]
        true_matches = [true_matches[idx] for idx in true_shuffle_ids]
        # for wrong matches, independently shuffle each view
        wrong_matches = features[subset_size:]
        random.shuffle(wrong_matches)
        # merge with true_ids
        features = []
        for i in range(dataset_size):
            if len(true_ids_temp) > 0 and i == true_ids_temp[0]:
                # insert true
                features.append(true_matches[0])
                true_ids_temp = true_ids_temp[1:]
                true_matches = true_matches[1:]
            else:
                # insert wrong
                features.append(wrong_matches[0])
                wrong_matches = wrong_matches[1:]
        assert len(true_matches) == 0 and len(wrong_matches) == 0, "match fill not exhausted"
        all_features[view] = features

    class_matches = {view: key[:num_matched_classes] for view, key in keys.items()}

    return all_features, true_ids, dataset_size, subset_size, nclasses, class_matches


def get_video_pair_data(args):
    path = Path(args.video_pair_data_path).resolve()
    paths = list(path.glob('*/*'))
    paths = {"{}_{}".format(p.parent.stem, p.stem): p for p in paths}
    data = {k: load_chunks(v) for k, v in paths.items()}
    data = {k: categorize_data(v) for k, v in data.items()}
    all_features, true_ids, dataset_size, subset_size, nclasses, class_matches = \
        get_derangements(data, deranged_classes_ratio=args.deranged_classes_ratio,
                         shuffle_true_ids=args.shuffle_true_ids,
                         class_datapoints_threshold=args.class_datapoints_threshold)
    all_features, labels = remove_labels(all_features)
    all_features = {view_key: [torch.Tensor(features) for features in view]
                    for view_key, view in all_features.items()}

    return all_features, true_ids, dataset_size, subset_size, nclasses, labels, class_matches
