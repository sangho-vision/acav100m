import time

from tqdm import tqdm

import numpy as np
import torch


DISTANCES = {
    'pca': 'inner_product',
    'pca_ip': 'inner_product',
    'pca_cs': 'cosine_similarity',
    'pca_l1': 'euclidean_diff_l1',
    'pca_l2': 'euclidean_diff_l2',
}


class PCAOptim:
    def __init__(self, pcas, measure_type='inner_product'):
        self.distance_type = measure_type.lower()
        distance_dt = {
            'inner_product': self.calc_inner_product,
            'cosine_similarity': self.calc_cosine_similarity,
            'euclidean_diff_l1': self.calc_euclidean_diff_l1,
            'euclidean_diff_l2': self.calc_euclidean_diff_l2,
        }
        assert self.distance_type in distance_dt, \
            "invalid distance type {}".format(self.distance_type)
        self._calc_distance = distance_dt[self.distance_type]
        self.pcas = pcas

    def calc_distances(self, pcas):
        distances = None
        count = 0
        for c1, c2 in self.combinations:
            distance = self.calc_distance(pcas[c1], pcas[c2])
            if distances is None:
                distances = distance
            else:
                distances += distance
            count += 1

        distances = distances / count  # mean over combinations
        return distances  # V x V

    def calc_distance(self, x1, x2):
        if isinstance(x1, np.ndarray):
            x1 = torch.from_numpy(x1)
        if isinstance(x2, np.ndarray):
            x2 = torch.from_numpy(x2)
        distance = self._calc_distance(x1, x2)  # V
        return distance

    def calc_inner_product(self, x1, x2):
        return torch.einsum('vc,vc->v', x1, x2)

    def calc_cosine_similarity(self, x1, x2):
        v, c = x1.shape
        return torch.nn.functional.cosine_similarity(x1, x2, dim=1)

    def calc_euclidean_diff(self, x1, x2, f=lambda x: x):
        diff = x1 - x2
        distance = f(diff)
        return -distance.sum(dim=-1)

    def calc_euclidean_diff_l1(self, x1, x2):
        return self.calc_euclidean_diff(x1, x2, f=lambda x: x.abs())

    def calc_euclidean_diff_l2(self, x1, x2):
        return self.calc_euclidean_diff(x1, x2, f=lambda x: (x ** 2))

    def init(self, clustering_combinations, candidates):
        self.combinations = clustering_combinations
        self.distances = self.calc_distances(self.pcas)  # list(D) x V x C -> V x V
        del self.pcas
        # V = self.distances.shape[0]

    '''
    def calc_score(self, candidates):
        # cand W
        row = self.row_cache.unsqueeze(0) + self.distances[candidates]  # W x V
        W, V = row.shape
        # W <= V
        eye = torch.cat((torch.eye(W)[:, :V], torch.zeros(W, V - W)), dim=-1)
        duplicates = (row * eye)
        row = row - duplicates  # W x V
        '''

    def calc_measure(self, subset_size):
        scores, ids = self.distances.topk(subset_size, sorted=True)  # V
        return scores, ids

    def run(self, subset_size, start_indices, intermediate_target=None,
            celf_ratio=0):
        S = start_indices  # ignore
        S = []
        GAIN = []
        LOOKUPS = []
        timelapse = []

        greedy_start_time = time.time()
        scores, ids = self.calc_measure(subset_size)
        start_time = time.time()
        # start from empty index
        pbar = tqdm(range(0, subset_size), desc='greedy iter')
        prev_score = 0
        for j in pbar:
            start_time = time.time()
            timelapse.append(time.time() - start_time)
            S.append(ids[j].item())
            current_score = scores[j].item()
            score = current_score + prev_score
            GAIN.append(score)
            prev_score = current_score
            LOOKUPS.append(0)  # greedy search renders num_lookups meaningless

            if intermediate_target is not None:
                precision = len(set(intermediate_target) & set(S)) / len(set(S))
                pbar.set_description("(LEN: {}, MEASURE: {}, PRECISION: {})".format(
                    len(S), score, precision))
            else:
                pbar.set_description("(LEN: {}, MEASURE: {})".format(len(S), score))

        tqdm.write("Time Consumed: {} seconds".format(time.time() - greedy_start_time))
        return (S, GAIN, timelapse, LOOKUPS)
