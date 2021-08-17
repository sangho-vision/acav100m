import copy
import time
from typing import List

from tqdm import tqdm

import torch
import numpy as np

from clustering import Clustering


# W: num_candidates, P: num_clustering_pairs, C: num_centroids
# N: (P x C X C), a: (P x C), b: (P x C), n: (P)
# WN: (W x P x C X C), Wa: (W x P x C), Wb: (W x P x C), wn: (W x P)


class EfficientMI:
    """ this implementation requires the users to use the same ncentroids for all clusterings """
    def __init__(self, clusterings: List[Clustering], measure_type='mutual_info',
                 average_method='arithmetic'):
        self.average_method = average_method.lower()
        self.clusterings = clusterings
        self.ncentroids = clusterings[0].ncentroids
        assignments = np.array([clustering.ind2cen for clustering in clusterings])  # D x V
        self.assignments = torch.from_numpy(assignments).to(torch.long).t()  # V x D
        self.eps = np.finfo('float64').eps

    def init(self, clustering_combinations, candidates):
        if isinstance(clustering_combinations, dict):
            self.combinations = clustering_combinations['pairing']
            self.pair_weights = clustering_combinations['weights']
        else:
            self.combinations = clustering_combinations
        self.init_cache()
        self.init_candidates(candidates)

    def get_N(self, assignments):
        C = self.ncentroids
        assignments = self.one_hot(assignments, C)  # W x D x C
        pair_ids = torch.LongTensor(self.combinations)  # P x 2
        assignments = assignments.to(self.device)
        pair_ids = pair_ids.to(self.device)
        p1 = self.gather_pairs(assignments, pair_ids[:, 0])
        p2 = self.gather_pairs(assignments, pair_ids[:, 1])  # W x P x C
        N = torch.einsum('wpa,wpb->wpab', p1, p2)  # B x P x C x C
        return N

    def get_assignments(self, candidates):
        candidates = torch.LongTensor(candidates)  # W
        assignments = self.assignments  # V x D
        assignments = assignments.index_select(dim=0, index=candidates)
        return assignments

    def init_cache(self):
        P = len(self.combinations)
        C = self.ncentroids
        N = torch.full((P, C, C), self.eps)
        a = N.sum(dim=1)
        b = N.sum(dim=2)
        n = a.sum(dim=-1)
        self.cache = {'N': N, 'a': a, 'b': b, 'n': n}

    def init_candidates(self, candidates):
        self.candidate_ids = torch.LongTensor(candidates)
        assignments = self.get_assignments(candidates)
        N = self.get_N(assignments)
        a = N.sum(2)
        b = N.sum(3)
        n = b.sum(-1)
        self.candidates = {'N': N, 'a': a, 'b': b, 'n': n}

    @staticmethod
    def gather_pairs(assignments, idx):
        W, _, C = assignments.shape
        idx = idx.unsqueeze(0).unsqueeze(-1)
        idx = idx.repeat(W, 1, C)
        return assignments.gather(dim=1, index=idx)  # W x P x C

    @staticmethod
    def one_hot(x, N, default=0, value=1):
        dtype = torch.float
        device = x.device
        x_onehot = torch.full((*x.shape, N), default, dtype=dtype).to(device)
        value = torch.full(x_onehot.shape, value, dtype=dtype).to(device)
        x_onehot.scatter_(dim=-1, index=x.unsqueeze(-1), src=value)
        return x_onehot

    def calc_score(self, *args, **kwargs):
        scores = self._calc_score(*args, **kwargs)
        scores = scores.mean(dim=-1)  # W
        score, idx = scores.max(dim=0)
        return score.item(), idx.item()

    def _calc_score(self, *args, **kwargs):
        scores = self.calc_MI(*args, **kwargs)
        if hasattr(self, 'pair_weights'):
            pair_weights = torch.tensor(self.pair_weights).float()  # P
            pair_weights = pair_weights.to(self.device)
            scores = torch.einsum('wp,p->wp', scores, pair_weights)
        return scores

    def calc_MI(self, last):
        N = last['N']  # W x P x C x C
        a = last['a'].unsqueeze(2)  # W x P x 1 x C
        b = last['b'].unsqueeze(3)
        n = last['n'].unsqueeze(-1).unsqueeze(-1)
        scores = (N / n * (N.log() + n.log() - (a.log() + b.log()))).sum([2, 3])  # W x P
        return scores

    def get_last(self, candidates=None):
        if candidates is None:
            candidates = self.candidates
        last = {key: self.cache[key].unsqueeze(0) + candidates[key]
                for key in candidates.keys()}
        return last

    def update_cache(self, last, idx):
        for key in last.keys():
            self.cache[key] = last[key][idx]

    def remove_idx_all(self, idx):
        self.remove_idx('candidate_ids', idx)
        self.remove_idx('candidates', idx)

    def calc_measure(self, celf=False):
        if celf:
            return self.calc_measure_celf()
        else:
            return self.calc_measure_greedy()

    def calc_measure_greedy(self):
        last = self.get_last()
        score, idx = self.calc_score(last)
        candidate_idx = self.candidate_ids[idx].item()
        self.update_cache(last, idx)
        self.remove_idx_all(idx)
        return score, candidate_idx, 1

    def calc_measure_celf(self):
        check, lookup = False, 0

        if hasattr(self, 'candidate_ids'):
            self.reverse_id_map = {v.item(): i for i, v in enumerate(list(self.candidate_ids))}
        while not check:
            lookup += 1
            current = self.Q_idx[0]
            current_gain, _ = self.calc_measure_single(current)
            current_diff = current_gain - self.gain
            self.Q_val[0] = current_diff
            # Q = sorted(Q, key=lambda x: x[1], reverse=True)
            self.Q_val, idx = self.Q_val.sort(descending=True)
            self.Q_idx = self.Q_idx.index_select(dim=0, index=idx)
            check = (self.Q_val[0] == current_diff)  # tie

        self.gain += self.Q_val[0]
        s = self.Q_idx[0].item()
        score = self.gain

        self.Q_val = self.Q_val[1:]
        self.Q_idx = self.Q_idx[1:]

        reversed_id = self.reverse_id_map[s]
        self.update_cache_celf(s)
        self.remove_idx_all(reversed_id)

        return score, s, lookup

    def init_celf_q(self, prev_score):
        self.Q_idx = copy.deepcopy(self.candidate_ids)
        last = self.get_last()
        scores = self._calc_score(last)
        scores = scores.mean(dim=-1)  # W
        self.Q_val = scores
        self.gain = prev_score

    def calc_measure_single(self, current):
        current = current.item()
        if hasattr(self, 'reverse_id_map'):
            idx = self.reverse_id_map[current]
        else:
            idx = current  # alignment
        candidate = {k: m[idx].unsqueeze(0) for k, m in self.candidates.items()}
        last = self.get_last(candidate)
        score, _ = self.calc_score(last)
        return score, idx

    def update_cache_celf(self, current):
        if hasattr(self, 'reverse_id_map'):
            idx = self.reverse_id_map[current]
        else:
            idx = current  # alignment
        candidate = {k: m[idx].unsqueeze(0) for k, m in self.candidates.items()}
        last = self.get_last(candidate)
        for key in last.keys():
            self.cache[key] = last[key][0]

    def remove_idx(self, name, idx):
        if hasattr(self, name):
            data = getattr(self, name)
            if isinstance(data, dict):
                data = {key: self._remove_idx(val, idx) for key, val in data.items()}
            else:
                data = self._remove_idx(data, idx)
            setattr(self, name, data)

    def _remove_idx(self, data, idx):
        return torch.cat((data[:idx], data[idx + 1:]), dim=0)

    def _add_samples(self, ids):
        '''
        assignments = torch.LongTensor([[c.get_assignment(x) for c in self.clusterings]
                                        for x in ids])  # W x D
        '''
        C = self.clusterings[0].ncentroids
        assignments = self.get_assignments(ids)
        N_whole = self.get_N(assignments)
        N = N_whole.sum(0)  # P x C x C
        a = N.sum(1)  # P x C
        b = N.sum(2)
        n = b.sum(-1)  # P
        return {'N': N, 'a': a, 'b': b, 'n': n}

    def add_samples(self, candidate_ids):
        to_add = self._add_samples(candidate_ids)
        for idx in candidate_ids:
            self.remove_idx_all(idx)
        for key in to_add.keys():
            self.cache[key] += to_add[key]
        '''
        cand_ids = [self.candidates[idx] for idx in ids]
        for idx in cand_ids:
            # DEBUG: variable array length!
            self.remove_idx_all(idx)
        '''

    def run_greedy(self, subset_size, start_indices, intermediate_target=None):
        return self.run(subset_size, start_indices, intermediate_target, celf_ratio=0)

    def run(self, subset_size, start_indices, intermediate_target=None, celf_ratio=0):
        # celf_ratio = 0 -> full greedy
        # greedy for the first (n_iters * (1 - celf_ratio)), celf for the rest
        assert celf_ratio >= 0 and celf_ratio <= 1, 'invalid celf_ratio {}'.format(celf_ratio)
        S = start_indices
        GAIN = []
        LOOKUPS = []
        timelapse = []

        self.add_samples(start_indices)

        greedy_start_time = time.time()
        start_time = time.time()
        # start from empty index
        iters = list(range(len(start_indices), subset_size - 1))
        niters = len(iters)
        greedy_niters = round(niters * (1 - celf_ratio))
        greedy_iters = iters[:greedy_niters]
        celf_iters = iters[greedy_niters:]
        celf_niters = len(celf_iters)
        print("niters: {} (greedy: {}, celf: {})".format(niters, greedy_niters,
                                                         celf_niters))

        pbar = tqdm(greedy_iters, desc='greedy iter')
        for j in pbar:
            start_time = time.time()
            score, idx, lookup = self.calc_measure(celf=False)
            timelapse.append(time.time() - start_time)
            S.append(idx)
            GAIN.append(score)
            LOOKUPS.append(lookup)

            if intermediate_target is not None:
                precision = len(set(intermediate_target) & set(S)) / len(set(S))
                pbar.set_description("(LEN: {}, MEASURE: {}, PRECISION: {})".format(
                    len(S), score, precision))
            else:
                pbar.set_description("(LEN: {}, MEASURE: {})".format(len(S), score))

        if len(celf_iters) > 0:
            prev_score = GAIN[-1] if len(GAIN) > 0 else 0
            self.init_celf_q(prev_score)
            pbar = tqdm(celf_iters, desc='celf iter')
            for j in pbar:
                start_time = time.time()
                score, idx, lookup = self.calc_measure(celf=True)
                timelapse.append(time.time() - start_time)
                S.append(idx)
                GAIN.append(score)
                LOOKUPS.append(lookup)

                if intermediate_target is not None:
                    precision = len(set(intermediate_target) & set(S)) / len(set(S))
                    pbar.set_description("(LEN: {}, MEASURE: {}, PRECISION: {})".format(
                        len(S), score, precision))
                else:
                    pbar.set_description("(LEN: {}, MEASURE: {})".format(len(S), score))

        tqdm.write("Time Consumed: {} seconds".format(time.time() - greedy_start_time))
        return (S, GAIN, timelapse, LOOKUPS)

    def ensure_nonzero(self, x):
        if torch.is_tensor(x):
            x = torch.max(x, torch.full(x.shape, self.eps, dtype=x.dtype))
        else:
            x = max(x, self.eps)
        return x

    def generalized_mean(self, ha, hb):
        if self.average_method == 'max':
            normalizer = torch.max(ha, hb)  # max avg
        elif self.average_method == 'min':
            normalizer = torch.min(ha, hb)
        else:
            # default is arithmetic
            normalizer = (ha + hb) / 2  # arithmetic mean
        return normalizer


class EfficientAMI(EfficientMI):
    """ adjusted MI """
    def _calc_score(self, *args, **kwargs):
        return self.calc_AMI(*args, **kwargs)

    def calc_EMI(self, last):
        # maybe sklearn.metrics.cluster.expected_mutual_information?
        # we need a way to 'DP' the factorials for faster computation
        N = last['N']  # W x P x C x C
        a = last['a'].unsqueeze(2)  # W x P x 1 x C
        b = last['b'].unsqueeze(3)
        n = last['n'].unsqueeze(-1).unsqueeze(-1)

        term1 = (N / n * (N.log() + n.log() - (a.log() + b.log())))
        log_term2 = (a + 1).lgamma() + (b + 1).lgamma() + (n - a + 1).lgamma() + (n - b + 1).lgamma() \
            - ((n + 1).lgamma() + (N + 1).lgamma() + (a - N + 1).lgamma() + (b - N + 1).lgamma()
               + (n - a - b + N + 1).lgamma())
        scores = (term1 * log_term2.exp()).sum([2, 3])
        return scores

    @staticmethod
    def calc_entropy(x, n):
        p = x / n  # W x P x C
        return -(p * p.log()).sum(dim=-1)

    def calc_entropies(self, last):
        a = last['a']  # W x P x C
        b = last['b']  # W x P x C
        n = last['n'].unsqueeze(-1)  # W x P x 1
        ha = self.calc_entropy(a, n)
        hb = self.calc_entropy(b, n)
        return ha, hb

    def calc_AMI(self, last):
        mi = self.calc_MI(last)
        emi = self.calc_EMI(last)
        ha, hb = self.calc_entropies(last)
        normalizer = self.generalized_mean(ha, hb)

        denominator = normalizer - emi
        '''
        if denominator < 0:
            denominator = min(denominator, -np.finfo('float64').eps)
        else:
            denominator = max(denominator, np.finfo('float64').eps)
        '''
        denominator = self.ensure_nonzero(denominator)
        ami = (mi - emi) / denominator
        return ami


class EfficientNMI(EfficientAMI):
    def _calc_score(self, *args, **kwargs):
        return self.calc_NMI(*args, **kwargs)

    def calc_NMI(self, last):
        mi = self.calc_MI(last)
        ha, hb = self.calc_entropies(last)
        normalizer = self.generalized_mean(ha, hb)
        normalizer = self.ensure_nonzero(normalizer)
        return (2 * mi) / normalizer


class ConstantMeasure(EfficientMI):
    def _calc_score(self, *args, **kwargs):
        return self.calc_constant(*args, **kwargs)

    def calc_constant(self, last):
        n = last['n']  # W x P
        return torch.full_like(n, 1)
