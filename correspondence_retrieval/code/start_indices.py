import random

from tqdm import tqdm

import numpy as np


def get_start_indices(clustering, nclasses, ntargets_per_class, option='random_one_per_class'):
    func_dict = {
        'random_one_per_class': get_random_one_per_class,
        'random_uniform_cluster': get_random_uniform_cluster,
        'zero': get_zero,
    }
    assert option in func_dict, "start indices method {} not implemented".format(option)
    tqdm.write("using start_indices selection method: {}".format(option))
    return func_dict[option](clustering, nclasses, ntargets_per_class)


def get_random_one_per_class(clustering, nclasses, ntargets_per_class):
    start_indices = [
        j * ntargets_per_class + random.randrange(ntargets_per_class)
        for j in range(nclasses)
    ]
    return start_indices


def get_random_uniform_cluster(clustering, nclasses, ntargets_per_class):
    # get one per cluster?
    views = sorted(list(clustering.keys()))
    view_idx_map = {v: i for i, v in enumerate(views)}
    ncentroids = np.array([clustering[view].ncentroids for i, view in enumerate(views)])
    argmax_view = ncentroids.argmax()
    ncentroids = clustering[views[argmax_view]].ncentroids
    start_indices = []
    assignments = np.full((len(views), ncentroids), 0)  # empty
    for i in range(ncentroids):
        view = views[argmax_view]  # pivot
        cluster_idx = (assignments[view_idx_map[view]] == 0).argmax(axis=0)  # get first empty
        current_cluster = clustering[view].cen2ind[cluster_idx]
        cluster_size = len(current_cluster)
        shuffle = np.arange(cluster_size)
        np.random.shuffle(shuffle)
        for i in range(len(shuffle)):
            idx = current_cluster[shuffle[i]]
            picks = []
            oks = []
            for j, view in enumerate(views):
                if j == argmax_view:
                    continue  # skip pivot
                cluster = clustering[view].get_assignment(idx)
                picks.append((j, cluster))
                if assignments[j][cluster] > 0 and i != (len(shuffle) - 1):  # fallback to default on last iteration
                    break
                oks.append(1)
            if len(oks) == len(views[1:]):  # ok
                break

        # pick current idx
        start_indices.append(idx)
        assignments[argmax_view][cluster_idx] += 1
        for i, cluster in picks:
            assignments[i][cluster] += 1  # mark filled

    assert len(start_indices) == ncentroids, "insufficient number of start_indices picked"
    assert assignments.sum() == ncentroids * len(views), "assignments value mismatch"

    assignment_variance = ((assignments - 1) ** 2).mean(axis=1)
    print("assignment_variance: {}".format(assignment_variance))

    return start_indices


def get_zero(clustering, nclasses, ntargets_per_class):
    '''
    len_clustering = len(clustering)
    return np.zeros((len_clustering, 1))
    '''
    return [0]
