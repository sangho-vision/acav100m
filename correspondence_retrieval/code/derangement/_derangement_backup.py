import copy
import random
from collections import defaultdict
from itertools import chain


def get_derangements(views_features, ntargets_per_class=3000, shuffle_true_ids=True):
    def random_derangement(n):
        while True:
            v = list(range(n))
            for j in range(n - 1, -1, -1):
                p = random.randint(0, j)
                if v[p] == j:
                    break
                else:
                    v[j], v[p] = v[p], v[j]
            else:
                if v[0] != 0:
                    return v

    def random_derangements(keys, n):
        derangements = {}
        for i, key in enumerate(keys):
            if i == 0:
                # do not perform derangement for the first view
                derangements[key] = list(range(n))
            else:
                derangements[key] = random_derangement(n)
        return derangements

    nclasses = len(views_features[0].keys())
    derangements = random_derangements(list(views_features.keys()), nclasses)
    true_matches = defaultdict(list)
    wrong_matches = defaultdict(list)
    for view_index, view_features in views_features.items():
        for key in view_features.keys():
            true_matches[view_index].append(views_features[view_index][key][:ntargets_per_class])
            wrong_match = views_features[view_index][derangements[view_index][key]][ntargets_per_class:]
            wrong_matches[view_index].append(wrong_match)

    num_true_samples = sum(x.shape[0] for x in true_matches[0])
    num_wrong_samples = sum(x.shape[0] for x in wrong_matches[0])
    subset_size = num_true_samples
    dataset_size = num_true_samples + num_wrong_samples
    true_shuffle_ids = list(range(subset_size))
    random.shuffle(true_shuffle_ids)
    if shuffle_true_ids:
        true_ids = sorted(random.sample(list(range(dataset_size)), subset_size))
    else:
        true_ids = list(range(num_true_samples))
    all_features = {view: list(chain(*[*true_matches[view], *wrong_matches[view]])) for view in true_matches.keys()}
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

    return all_features, true_ids, subset_size
