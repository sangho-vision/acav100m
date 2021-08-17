import numpy as np


def get_entropy(m, eps=1e-20):
    ids = list(range(len(m.shape)))[1:]
    return -(m * np.log(m + eps)).mean(axis=tuple(ids))


def compare_classes_to_clusters(clusterings, labels, class_matches):
    '''
    for subset in ['matched', 'unmatched']:
        print(subset)
        _compare_classes_to_clusters(clusterings, labels, class_matches, subset)
    '''
    view = 'HMDB51_slow_fast'
    labels = labels[view]
    clustering = clusterings[view]
    ncentroids = clustering.ncentroids

    set_labels = sorted(list(set(labels)))
    categories = {v: i for i, v in enumerate(set_labels)}
    matches = sorted([categories[x] for x in class_matches[view]])
    labels = [categories[x] for x in labels]  # to idx
    label_cen = np.zeros((len(categories), ncentroids), dtype=float)
    for i, label in enumerate(labels):
        cen = clustering.get_assignment(i)
        label_cen[label, cen] += 1
    import ipdb; ipdb.set_trace()  # XXX DEBUG


def compare_clusterings_pairwise(clusterings, labels, class_matches):
    views_labels = labels
    outputs = {}
    true_matches = {}
    ent_outputs = {}
    ent_true_matches = {}
    ent_false_matches = {}
    out_views = sorted(list(views_labels.keys()))
    for out_view in out_views:
        labels = views_labels[out_view]
        views = sorted(list(clusterings.keys()))
        ncentroids = clusterings[views[0]].ncentroids
        categories = {v: i for i, v in enumerate(set(labels))}
        matches = sorted([categories[x] for x in class_matches[out_view]])
        labels = [categories[x] for x in labels]  # to idx
        label_cen2cen = np.zeros((len(categories), *([ncentroids] * len(views))), dtype=float)
        for i, label in enumerate(labels):
            cens = []
            for view in views:
                cen = clusterings[views[0]].get_assignment(i)
                cens.append(cen)
            label_cen2cen[tuple([label, *cens])] += 1
        outputs[out_view] = global_normalize(label_cen2cen, exclude_dim=0)
        ent_outputs[out_view] = get_entropy(outputs[out_view])
        true_match = label_cen2cen[matches]
        true_matches[out_view] = global_normalize(true_match, exclude_dim=0)
        ent_true_matches[out_view] = get_entropy(true_matches[out_view])
        false_match_classes = [i for v, i in categories.items() if v not in class_matches[out_view]]
        ent_false_matches[out_view] = ent_outputs[out_view][false_match_classes]

        print(out_view)
        print(ent_true_matches[out_view])
        print(ent_true_matches[out_view].mean())
        print(ent_false_matches[out_view])
        print(ent_false_matches[out_view].mean())

    # ent: 0 ~ 0.003, lower -> sparser
    import ipdb; ipdb.set_trace()  # XXX DEBUG


def _compare_classes_to_clusters(clusterings, labels, class_matches, subset):
    label_to_cens = []
    for view in clusterings.keys():
        label_to_cen = compare(clusterings[view], labels[view], class_matches[view], subset=subset)
        label_to_cens.append(label_to_cen)
    compare_assignments(*label_to_cens)  # only 2 views


def compare(clustering, labels, matched_classes, subset='matched'):
    if subset == 'all':
        return _compare(clustering, labels, set(labels))
    elif subset == 'matched':
        return _compare(clustering, labels, set(matched_classes))
    elif subset == 'unmatched':
        return _compare(clustering, labels, set(labels) - set(matched_classes))
    else:
        return None


def _compare(clustering, labels, set_labels):
    set_labels = sorted(list(set_labels))
    categories = {v: i for i, v in enumerate(set_labels)}
    labels = [categories[x] if x in categories else None for x in labels]  # to idx
    cen_to_label = np.zeros((clustering.ncentroids, len(categories)), dtype=float)
    for i, label in enumerate(labels):
        if label is not None:
            cen = clustering.get_assignment(i)
            cen_to_label[cen, label] += 1

    '''
    # only works for set(labels)
    for cen in range(clustering.ncentroids):
        len_orig = len(clustering.get_cluster(cen))
        len_added = cen_to_label[cen].sum()
        assert len_orig == len_added, \
            "clustering size mismatch in {}: {}, {}".format(cen, len_orig, len_added)
    '''
    label_to_cen = cen_to_label.transpose(0, 1)
    '''
    with np.printoptions(threshold=np.inf):
        # print(label_to_cen)
        print(normalize(label_to_cen, dim=1))
    '''
    return normalize(label_to_cen, dim=1)


def normalize(x, dim=0, eps=1e-10):
    return x / (x.sum(axis=dim, keepdims=True) + eps)


def global_normalize(x, exclude_dim=0, eps=1e-10):
    ids = list(range(len(x.shape)))
    ids.remove(exclude_dim)
    ids = tuple(ids)
    x_sum = x.sum(axis=ids)
    x_sum = np.expand_dims(x_sum, axis=ids)
    return x / (x_sum + eps)


def compare_assignments(view1, view2):
    comp = view1[:, :, None] - view2[:, None, :]
    comp = comp ** 2
    comp = comp.mean(axis=0)
    closest_idx = comp.argmin(axis=-1)
    # comp_best = comp.min(axis=-1)
    comp = np.stack((view1, view2[:, closest_idx]), axis=-1)
    comp_score = ((comp[:, 0] - comp[:, 1]) ** 2).mean(axis=-1)
    print(comp_score)
    '''
    with np.printoptions(threshold=np.inf):
        print(comp)
    '''
