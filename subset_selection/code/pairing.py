from itertools import combinations, product
from collections import defaultdict


def get_cluster_pairing(keys, cluster_pairing):
    pairing_dict = {
        'diagonal': get_diagonal,
        'bipartite': get_bipartite,
        'combination': get_combination,
    }
    cluster_pairing = cluster_pairing.lower()
    assert cluster_pairing in pairing_dict, f"invalid cluster pairing type: {cluster_pairing}"
    return pairing_dict[cluster_pairing](keys)


def get_combination(keys):
    clustering_indices = range(len(keys))
    group_size = 2
    clustering_combinations = combinations(clustering_indices, group_size)
    return list(clustering_combinations)


def get_bipartite(keys):
    keys = {v: i for i, v in enumerate(keys)}
    views = defaultdict(list)
    for key, idx in keys.items():
        view = key[0]  # get dataset+model name
        views[view].append(idx)
    views = list(views.values())
    return list(product(*views))


def get_diagonal(keys):
    keys = {v: i for i, v in enumerate(keys)}
    clustering_names = defaultdict(list)
    for key, idx in keys.items():
        view = key[0]  # get dataset+model name
        clustering_name = key[1]
        clustering_names[clustering_name].append(idx)
    pairs = list(clustering_names.values())
    return pairs
