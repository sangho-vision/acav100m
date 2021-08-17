from typing import List
from itertools import combinations
from collections import defaultdict

from clustering import Clustering


class CustomMeasure(object):
    def __init__(self, clusterings: List[Clustering]):
        self.nclusterings = len(clusterings)
        self.clusterings = clusterings
        self.clustering_agreed = [
            clustering.get_num_agreed_pairs() for clustering in clusterings
        ]
        self.pair_assignments = defaultdict(lambda: None)

    def add_clustering(self, clustering: Clustering):
        self.nclusterings = self.nclusterings + 1
        self.clusterings.append(clustering)
        self.clustering_agreed.append(clustering.get_num_agreed_pairs())

    def get_clustering(self, ind):
        return self.clusterings[ind]

    def get_or_calc_pair_assignments(self, pair, clustering_indices):
        key = self.build_key(pair, clustering_indices)
        is_agreed = self.get_pair_assignments(key)
        if is_agreed is None:
            is_agreed = self.calc_pair_assignments(pair, clustering_indices)
        self.put_pair_assignments(key, is_agreed)
        return is_agreed

    def build_key(self, pair, clustering_indices):
        key1 = str(sorted(pair))
        key2 = str(sorted(clustering_indices))
        key = f"{key1}_{key2}"
        return key

    def get_pair_assignments(self, key):
        return self.pair_assignments[key]

    def put_pair_assignments(self, key, is_agreed):
        self.pair_assignments[key] = is_agreed

    def calc_pair_assignments(self, pair, clustering_indices):
        agreed = []
        for clustering_ind in clustering_indices:
            clustering = self.clusterings[clustering_ind]
            agreed.append(
                clustering.get_assignment(pair[0]) == clustering.get_assignment(pair[1])
            )
        is_agreed = all(agreed)
        return is_agreed

    def get_num_agreed_pairs(self, indices, clustering_indices, only_last=False):
        num_agreed = 0
        if only_last:
            ind_combinations = zip(
                indices[:-1], indices[-1:] * (len(indices) - 1)
            )
        else:
            ind_combinations = combinations(indices, 2)
        for pair in ind_combinations:
            is_agreed = self.get_or_calc_pair_assignments(pair, clustering_indices)
            if is_agreed:
                num_agreed += 1
        return num_agreed

    def get_combination(self):
        clustering_indices = range(self.nclusterings)
        # group_size = len(clustering_indices)
        group_size = 2
        clustering_combinations = combinations(clustering_indices, group_size)
        return clustering_combinations

    def get_measure(self, indices, clustering_combinations=None, agreed_dict={}):
        if clustering_combinations is None:
            clustering_combinations = self.get_combination()
        new_agreed_dict = {}
        measures = []
        for tuple_indices in clustering_combinations:
            only_last = tuple_indices in agreed_dict
            num_subset_agreed = self.get_num_agreed_pairs(
                indices, tuple_indices, only_last
            )
            if only_last:
                num_subset_agreed += agreed_dict[tuple_indices]
            new_agreed_dict[tuple_indices] = num_subset_agreed
            scores = [
                num_subset_agreed / self.clustering_agreed[ind]
                for ind in tuple_indices
            ]
            measures.append(sum(scores) / len(scores))
        measure = sum(measures) / len(measures)

        return measure, new_agreed_dict

    def __call__(self, *args, **kwargs):
        return self.get_measure(*args, **kwargs)
