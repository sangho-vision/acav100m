from typing import List
from itertools import combinations
from collections import defaultdict

import numpy as np
import sklearn.metrics as metrics

from clustering import Clustering


MEASURES = {
    "adjusted_rand": metrics.adjusted_rand_score,
    "fowlkes_mallows": metrics.fowlkes_mallows_score,
    "mutual_info": metrics.mutual_info_score,
    "adjusted_mutual_info": metrics.adjusted_mutual_info_score, # equal to Normalized Variation of Information
    "normalized_mutual_info": metrics.normalized_mutual_info_score,
}


class MutualInformation(object):
    """ mean of pairwise mutual information """
    def __init__(self, clusterings: List[Clustering], measure_type='mutual_info'):
        self.nclusterings = len(clusterings)
        self.clusterings = clusterings
        self.measure_type = measure_type
        self.measure = MEASURES[self.measure_type]
        # self.measure_dict = defaultdict(lambda: 0)

    '''
    def init_count_dict(self):
        clustering_indices = range(self.nclusterings)  # do not use clustering_indices
        clustering_combinations = combinations(clustering_indices, 2)
        count_dict = {}
        for tuple_indices in clustering_combinations:
            idx = sorted(tuple_indices)
            m = np.full((self.get_clustering(idx[0]).ncentroids,
                        self.get_clustering(idx[1]).ncentroids), self.eps)
            dict_key = self.get_dict_key(tuple_indices)
            count_dict[dict_key] = m
        return count_dict
    '''

    def get_clustering(self, ind):
        return self.clusterings[ind]

    def get_assignment(self, indices, cluster_ind):
        clustering = self.clusterings[cluster_ind]
        return [clustering.get_assignment(ind) for ind in indices]

    '''
    def get_dict_key(self, tuple_indices):
        key = tuple(tuple_indices)
        key = str(sorted(key))
        return key

    def update_count(self, count_dict, tuple_indices, indices):
        tuple_indices = sorted(tuple_indices)
        dict_key = self.get_dict_key(tuple_indices)
        for idx in indices:
            c1 = self.get_clustering(tuple_indices[0])
            c2 = self.get_clustering(tuple_indices[1])
            a1 = c1.get_assignment(idx)
            a2 = c2.get_assignment(idx)
            count_dict[dict_key][a1, a2] += 1
        return count_dict
    '''

    def get_combination(self):
        clustering_indices = range(self.nclusterings)
        group_size = 2
        clustering_combinations = combinations(clustering_indices, group_size)
        return clustering_combinations

    def get_measure(self, indices, clustering_combinations=None, agreed_dict={}):
        if clustering_combinations is None:
            clustering_combinations = self.get_combination()
        new_agreed_dict = {}
        measures = []
        for pair_indices in clustering_combinations:
            indices1 = self.get_assignment(indices, pair_indices[0])
            indices2 = self.get_assignment(indices, pair_indices[1])
            measures.append(self.measure(indices1, indices2))
        measure = sum(measures) / len(measures)

        return measure, new_agreed_dict

    '''
    def calc_measure(self, count_dict, cluster_pair):
        cluster_pair = sorted(cluster_pair)
        dict_key = str(cluster_pair)
        n = count_dict[dict_key]
        N = n.sum()
        a = n.sum(axis=0, keepdims=True)
        b = n.sum(axis=1, keepdims=True)
        measure = self.calc_func(n, a, b, N)
        return measure

    @staticmethod
    def calc_MI(n, a, b, N):
        return (n / N * (np.log(n * N) - np.log(a * b))).sum()
    '''

    def __call__(self, *args, **kwargs):
        return self.get_measure(*args, **kwargs)
