import random
import copy
from collections import defaultdict

from utils import split_dataset_model_names, merge_dataset_model_name


def categorize_data_view(data, add_vid=False, align=False):
    if not align:
        return {k: categorize_data(v, add_vid) for k, v in data.items()}
    else:
        views, _ = match_datapoints(data, add_vid=True)
        res = {}
        for k, view in views.items():
            classes = defaultdict(list)
            for datum in view:
                # vid = datum['vid']
                if not add_vid:
                    datum.pop('vid')
                classes[datum['label']].append(datum)
            res[k] = dict(classes)
        return res


def categorize_data(data, add_vid=False):
    classes = defaultdict(dict)
    for vid, datum in data.items():
        if add_vid:
            datum = {**datum, 'vid': vid}
        classes[datum['label']][vid] = datum
    for label, class_dt in classes.items():
        keys = sorted(list(class_dt.keys()))
        classes[label] = [class_dt[key] for key in keys]
    return classes


def match_datapoints(views, add_vid=False):
    unique_keys = None
    for view, features in views.items():
        if unique_keys is None:
            unique_keys = set(features.keys())
        else:
            unique_keys = unique_keys & set(features.keys())
    # fix order to align indices
    unique_keys = sorted(list(unique_keys))
    dataset_size = len(unique_keys)
    for view, features in views.items():
        views[view] = [features[k] for k in unique_keys]
        if add_vid:
            views[view] = [{**features[k], 'vid': k} for k in unique_keys]
        else:
            views[view] = [features[k] for k in unique_keys]
    return views, dataset_size


def get_shuffled_ids(views, dataset_size, subset_size, shuffle_true_ids=True):
    # randomness involved
    true_shuffle_ids = list(range(subset_size))
    random.shuffle(true_shuffle_ids)
    if shuffle_true_ids:
        print("shuffled true_ids")
        true_ids = random.sample(list(range(dataset_size)), subset_size)
    else:
        true_ids = list(range(subset_size))
    wrong_shuffle_ids = {view: list(range(dataset_size - subset_size)) for view in views.keys()}
    for wrong_shuffle_idx in wrong_shuffle_ids.values():
        random.shuffle(wrong_shuffle_idx)
    return true_shuffle_ids, wrong_shuffle_ids, true_ids


def wrapped_run_derangements(all_features, keys, dataset_size, subset_size,
                             num_matched_classes, nclasses,
                             true_shuffle_ids, wrong_shuffle_ids, true_ids):
    res = _run_derangements(all_features, dataset_size, subset_size,
                            true_shuffle_ids, wrong_shuffle_ids, true_ids)
    class_matches = {view: key[:num_matched_classes] for view, key in keys.items()}
    return (*res, nclasses, class_matches)


def _run_derangements(all_features, dataset_size, subset_size,
                      true_shuffle_ids, wrong_shuffle_ids, true_ids):
    true_ids = sorted(true_ids)
    for view, features in all_features.items():
        true_ids_temp = copy.deepcopy(true_ids)
        # for true matches, align all views
        true_matches = features[:subset_size]
        true_matches = [true_matches[idx] for idx in true_shuffle_ids]
        # for wrong matches, independently shuffle each view
        wrong_matches = features[subset_size:]
        wrong_matches = [wrong_matches[idx] for idx in wrong_shuffle_ids[view]]
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

    return all_features, true_ids, dataset_size, subset_size


def shuffle_each_view(view_classes, num_class_datapoints, shuffle_datapoints):
    view_classes_shuffled = {}
    view_layers = split_dataset_model_names(list(view_classes.keys()))  # dict of view: layer
    if shuffle_datapoints:
        # shuffle datapoints within each view
        view_keys = sorted(list(view_layers.keys()))
        for view_name in view_keys:
            indices = list(range(num_class_datapoints))
            random.shuffle(indices)
            layers = view_layers[view_name]
            for layer in layers:
                view = merge_dataset_model_name(view_name, layer)
                view_classes_shuffled[view] = [view_classes[view][i] for i in indices]
        assert set(view_classes.keys()) == set(view_classes_shuffled.keys()), \
            "view_classes keys changed after shuffling, likely to be dataset_model_names error"
    else:
        indices = list(range(num_class_datapoints))
        random.shuffle(indices)
        for view in view_classes.keys():
            # view-aligned shuffle
            view_classes_shuffled[view] = [view_classes[view][i] for i in indices]
    return view_classes_shuffled
