from itertools import combinations, product
from collections import defaultdict

from pair_weights import get_weights


def get_cluster_pairing(keys, cluster_pairing, weight_type=None):
    pairing_dict = {
        'diagonal': get_diagonal,
        'bipartite': get_bipartite,
        'combination': get_combination,
        'penultimate': get_penultimate
    }
    layer_names = {'layer_{}'.format(i): i for i in range(5)}
    cluster_pairing = cluster_pairing.lower()
    if cluster_pairing in layer_names:
        pairing = get_single_layer(keys, layer_names[cluster_pairing])
    else:
        assert cluster_pairing in pairing_dict, f"invalid cluster pairing type: {cluster_pairing}"
        pairing = pairing_dict[cluster_pairing](keys)
    return get_weights(keys, pairing, weight_type)


def get_single_layer(keys, layer=-1):
    # get aligned single layer features (before classification linear projection)
    keys = {v: i for i, v in enumerate(keys)}
    clustering_names = defaultdict(list)
    for key, idx in keys.items():
        view = key[:key.find('_')]  # get dataset+model name
        clustering_name = key[key.find('_') + 1:]
        clustering_names[clustering_name].append(idx)
    key = sorted(list(clustering_names.keys()))[layer]
    pairs = [clustering_names[key]]
    return pairs


def get_penultimate(keys):
    return get_single_layer(keys, layer=4)


def get_combination(keys):
    clustering_indices = range(len(keys))
    group_size = 2
    clustering_combinations = combinations(clustering_indices, group_size)
    return list(clustering_combinations)


def get_bipartite(keys):
    keys = {v: i for i, v in enumerate(keys)}
    views = defaultdict(list)
    for key, idx in keys.items():
        view = key[:key.find('_')]  # get dataset+model name
        # clustering_name = key[key.find('_') + 1:]
        views[view].append(idx)
    views = list(views.values())
    return list(product(*views))


def get_diagonal(keys):
    keys = {v: i for i, v in enumerate(keys)}
    clustering_names = defaultdict(list)
    for key, idx in keys.items():
        view = key[:key.find('_')]  # get dataset+model name
        clustering_name = key[key.find('_') + 1:]
        clustering_names[clustering_name].append(idx)
    pairs = list(clustering_names.values())
    return pairs
