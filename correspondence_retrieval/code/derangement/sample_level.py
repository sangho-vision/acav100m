import math
import random

from .common import get_shuffled_ids, _run_derangements, match_datapoints
from .derangement import get_keys, cut_class_datapoints
from .sharded_derangement import get_shards


'''
def get_sample_level_derangements(*args, **kwargs):
    return {'unsharded': _get_sample_level_derangements(*args, **kwargs)}
'''


def get_sample_level_sharded_derangements(*args, num_shards=10, **kwargs):
    all_features, true_ids, dataset_size, subset_size, nclasses, class_matches = \
        _get_sample_level_derangements(*args, **kwargs)
    res = {'unsharded': (all_features, true_ids, dataset_size, subset_size, nclasses, class_matches)}
    if num_shards is not None:
        # get shards
        sharded, sharded_ids = get_shards(all_features, true_ids, dataset_size,
                                          subset_size, nclasses, class_matches,
                                          num_shards)
        res['sharded'] = sharded
        res['sharded_ids'] = sharded_ids
    return res


def _get_sample_level_derangements(views, deranged_classes_ratio=0.5, shuffle_true_ids=True,
                                   class_datapoints_threshold=None,
                                   shuffle_datapoints=True,
                                   shuffle_each_cluster=False):
    '''
    views = {k: categorize_data(v) for k, v in views.items()}
    keys, nclasses, num_matched_classes = get_keys(views, deranged_classes_ratio,
                                                   shuffle_each_cluster,
                                                   if_shuffle_classes=False)
    views, keys, dataset_size, subset_size, num_matched_classes, nclasses = \
        cut_class_datapoints(views, class_datapoints_threshold,
                             keys, nclasses, num_matched_classes, shuffle_datapoints=False,
                             if_shuffle_each_view=False, fix_class_sizes=True)
    '''
    deranged_samples_ratio = deranged_classes_ratio
    nclasses = get_class_num(views)
    views, pre_cut_size = match_datapoints(views)
    all_datapoints_threshold = None
    if class_datapoints_threshold is not None:
        all_datapoints_threshold = math.floor(class_datapoints_threshold * nclasses)
    views, dataset_size, subset_size = shuffle_sample_level(views, pre_cut_size, all_datapoints_threshold,
                                                            deranged_samples_ratio)
    class_matches = None

    true_shuffle_ids, wrong_shuffle_ids, true_ids = \
        get_shuffled_ids(views, dataset_size, subset_size, shuffle_true_ids)
    all_features, true_ids, dataset_size, subset_size = \
        _run_derangements(views, dataset_size, subset_size,
                          true_shuffle_ids, wrong_shuffle_ids, true_ids)

    return all_features, true_ids, dataset_size, subset_size, nclasses, class_matches


def get_class_num(views):
    lens = []
    for view, features in views.items():
        classes = set()
        for vid, datum in features.items():
            classes.add(datum['label'])
        lens.append(len(classes))
    assert all(x == lens[0] for x in lens), 'class num difference between views'
    return lens[0]


def shuffle_sample_level(views, pre_cut_size, threshold, deranged_samples_ratio):
    if threshold > pre_cut_size:
        print(f"number of samples in the original set ({pre_cut_size}) is smaller than threshold ({threshold})")
        print(f"defaulting to use all samples ({pre_cut_size})")
    size = min(pre_cut_size, threshold)
    dataset_size = size
    subset_size = math.floor(dataset_size * deranged_samples_ratio)
    # balancing subset_size in case of deranged_samples_ratio=0.5
    dataset_size = round(subset_size / deranged_samples_ratio)
    indices = list(range(pre_cut_size))
    random.shuffle(indices)
    indices = indices[:size]
    return {view: [features[i] for i in indices] for view, features in views.items()}, \
        size, subset_size
