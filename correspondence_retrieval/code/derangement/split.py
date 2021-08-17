from collections import defaultdict

from utils import exchange_levels, merge_dicts, peek
from .common import categorize_data
from .derangement import match_classes_with_shuffle


def split_dataset(views, train_ratio, sample_level):
    train = {}
    test = defaultdict(dict)
    count = 0
    # add vid
    views = {view: {k: {**v2, 'vid': k} for k, v2 in v.items()} for view, v in views.items()}
    views = match_classes(views, sample_level)
    feature_classes = peek(views)
    for label, class_li in feature_classes.items():
        length = len(class_li)
        count += length
        train_len = round(length * train_ratio)
        class_trains = []
        keys = []
        for feature_name, feature in views.items():
            class_li = feature[label]
            # class_train = {row['vid']: row for row in class_li[:train_len]}
            class_train = class_li[:train_len]
            keys.append(feature_name)
            class_trains.append(class_train)
        train[label] = [dict(zip(keys, v)) for v in zip(*class_trains)]
        for feature_name, feature in views.items():
            class_li = feature[label]
            class_test = class_li[train_len:]
            class_test = {row['vid']: row for row in class_test}
            test[feature_name] = {**test[feature_name], **class_test}
    train_len = sum(len(v) for v in train.values())
    test_len = len(peek(test))
    assert train_len + test_len == count, "error in dataset splitting"
    return train, test, train_len, test_len


def match_classes(views, sample_level):
    all_features, keys, dataset_size, subset_size, num_matched_classes, nclasses = \
        match_classes_with_shuffle(views, 0, None, False, False,
                                   return_class_dict=True, add_vid=True, align=sample_level,
                                   if_shuffle_each_view=False,
                                   if_shuffle_classes=True)
    return all_features


def remove_vid(x):
    x.pop('vid')
    return x
