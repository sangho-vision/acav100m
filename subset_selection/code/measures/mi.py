import time

from tqdm import tqdm

import torch
import numpy as np


# assignments: V x D
# D: num_clusterings, V: num_total_set
# W: num_candidates, P: num_clustering_pairs, C: num_centroids


class EfficientMI:

    """ this implementation requires the users to use the same ncentroids for all clusterings """
    # N: (P x C X C), a: (P x C), b: (P x C), n: (P)
    # WN: (W x P x C X C), Wa: (W x P x C), Wb: (W x P x C), wn: (W x P)

    def __init__(self, assignments, measure_type='mutual_info',
                 average_method='arithmetic', ncentroids=20, **kwargs):
        self.average_method = average_method.lower()
        self.ncentroids = ncentroids
        self.assignments = torch.from_numpy(assignments).to(torch.long)  # V x D
        self.eps = np.finfo('float64').eps

    def init(self, clustering_combinations, candidates):
        self.combinations = clustering_combinations
        self.init_cache()
        self.init_candidates(candidates)

    def init_cache(self):
        P = len(self.combinations)
        C = self.ncentroids
        N = torch.full((P, C, C), self.eps)
        a = N.sum(dim=1)
        b = N.sum(dim=2)
        n = a.sum(dim=-1)
        self.cache = {'N': N, 'a': a, 'b': b, 'n': n}

    def get_assignments(self, candidates):
        candidates = torch.LongTensor(candidates)  # W
        assignments = self.assignments  # V x D
        assignments = assignments.index_select(dim=0, index=candidates)
        return assignments

    def init_candidates(self, candidates):
        self.candidate_ids = torch.LongTensor(candidates)
        assignments = self.get_assignments(candidates)
        C = self.ncentroids
        assignments = self.one_hot(assignments, C)  # W x D x C
        pair_ids = torch.LongTensor(self.combinations)  # P x 2
        p1 = self.gather_pairs(assignments, pair_ids[:, 0])
        p2 = self.gather_pairs(assignments, pair_ids[:, 1])
        N = torch.einsum('wpa,wpb->wpab', p1, p2)  # W x P x C x C
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
        x_onehot = torch.full((*x.shape, N), default, dtype=dtype)
        value = torch.full(x_onehot.shape, value, dtype=dtype)
        x_onehot.scatter_(dim=-1, index=x.unsqueeze(-1), src=value)
        return x_onehot

    def calc_score(self, *args, **kwargs):
        scores = self._calc_score(*args, **kwargs)
        scores = scores.mean(dim=-1)  # W
        score, idx = scores.max(dim=0)
        return score.item(), idx.item()

    def _calc_score(self, *args, **kwargs):
        return self.calc_MI(*args, **kwargs)

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

    def calc_measure(self):
        last = self.get_last()
        score, idx = self.calc_score(last)
        candidate_idx = self.candidate_ids[idx].item()
        self.update_cache(last, idx)
        self.remove_idx_all(idx)
        return score, candidate_idx

    def remove_idx(self, name, idx):
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
        assignments = self.get_assignments(ids)
        C = self.ncentroids
        assignments = self.one_hot(assignments, C)  # W x D x C
        pair_ids = torch.LongTensor(self.combinations)  # P x 2
        p1 = self.gather_pairs(assignments, pair_ids[:, 0])
        p2 = self.gather_pairs(assignments, pair_ids[:, 1])
        N_whole = torch.einsum('wpa,wpb->wpab', p1, p2)
        N = N_whole.sum(0)  # P x C x C
        a = N.sum(1)  # P x C
        b = N.sum(2)
        n = b.sum(-1)  # P
        return {'N': N, 'a': a, 'b': b, 'n': n}

    def add_samples(self, ids):
        to_add = self._add_samples(ids)
        for key in to_add.keys():
            self.cache[key] += to_add[key]

    def run_greedy(self, subset_size, start_indices, intermediate_target=None,
                   verbose=False, log_every=1, log_times=None,
                   node_rank=None, pid=None):
        S = start_indices
        GAIN = []
        LOOKUPS = []
        timelapse = []

        greedy_start_time = time.time()
        start_time = time.time()
        # start from empty index
        pbar = range(len(start_indices), subset_size - 1)
        niters = subset_size - 1 - len(start_indices)
        if log_times is not None:
            log_every = niters // log_times

        if verbose:
            pbar = tqdm(pbar, desc='greedy iter')
        for j in pbar:
            start_time = time.time()
            score, idx = self.calc_measure()
            timelapse.append(time.time() - start_time)
            S.append(idx)
            GAIN.append(score)
            LOOKUPS.append(0)  # greedy search renders num_lookups meaningless

            if verbose:
                if (j + 1) % log_every == 0:
                    if intermediate_target is not None:
                        precision = len(set(intermediate_target) & set(S)) / len(set(S))
                        msg = "(LEN: {}, MEASURE: {}, PRECISION: {})".format(
                            len(S), score, precision)
                    else:
                        msg = "(LEN: {}, MEASURE: {})".format(len(S), score)
                    if node_rank is not None:
                        msg = 'Node: {}, '.format(node_rank) + msg
                    if pid is not None:
                        msg = 'PID: {}, '.format(node_rank) + msg
                    pbar.set_description(msg)

        if verbose:
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


class EfficientMemMI(EfficientMI):
    def calc_N(self, candidates, save_ids=True):
        if save_ids:
            self.candidate_ids = torch.LongTensor(candidates)
        assignments = self.get_assignments(candidates)  # W x D
        pair_ids = torch.LongTensor(self.combinations)  # P x 2
        N = self.gather_pairs(assignments, pair_ids)  # W x P x 2
        return N

    def init_candidates(self, candidates):
        N = self.calc_N(candidates)
        self.candidates = {'N': N}

    def init_cache(self):
        super().init_cache()

        # P = len(self.combinations)
        N = self.cache['N']
        NlogN = (N * N.log()).sum([-1, -2])
        a = self.cache['a']
        aloga = (a * a.log()).sum(-1)
        b = self.cache['b']
        blogb = (b * b.log()).sum(-1)
        self.cache = {**self.cache,
                      'NlogN': NlogN, 'aloga': aloga, 'blogb': blogb}

    @staticmethod
    def gather_pairs(assignments, idx):
        # W x D, P x 2 -> W x P x 2
        W = assignments.shape[0]
        N = idx.shape[-1]
        P, N = idx.shape
        idx = idx.unsqueeze(0)
        idx = idx.repeat(W, 1, 1)  # W x P x 2
        assignments = assignments.unsqueeze(-1)
        assignments = assignments.repeat(1, 1, N)
        return assignments.gather(dim=1, index=idx)  # W x P x 2

    def get_last(self, candidates=None):
        if candidates is None:
            candidates = self.candidates['N']
        N = self.get_last_N(self.cache['N'], candidates)  # W x P
        a = self.get_last_ab(self.cache['a'], candidates, dim=1)  # W x P
        b = self.get_last_ab(self.cache['b'], candidates, dim=0)  # W x P
        last = {}
        last['NlogN'] = self.update_nlogn(self.cache['NlogN'].unsqueeze(0), N)
        last['aloga'] = self.update_nlogn(self.cache['aloga'].unsqueeze(0), a)
        last['blogb'] = self.update_nlogn(self.cache['blogb'].unsqueeze(0), b)
        last['n'] = (self.cache['n'] + 1).unsqueeze(0)
        return last

    @staticmethod
    def nlogn(x):
        return x * x.log()

    def update_nlogn(self, prev, num):
        return prev - self.nlogn(num) + self.nlogn(num + 1)

    def get_last_N(self, cache, candidates):
        # P x C x C, W x P x 2 -> W x P
        c1 = candidates[:, :, 0]
        cache = self.small_gather(cache, c1.t(), batch_dim=0).transpose(0, 1)  # W x P x C
        c2 = candidates[:, :, 1]
        cache = cache.gather(dim=-1, index=c2.unsqueeze(-1)).squeeze(-1)  # W x P
        return cache

    def get_last_ab(self, cache, candidates, dim=0):
        # P x C, W x P -> W x P
        c = candidates[:, :, dim]
        W, P = c.shape
        cache = cache.unsqueeze(0).repeat(W, 1, 1)
        cache = cache.gather(dim=-1, index=c.unsqueeze(-1)).squeeze(-1)
        return cache

    @staticmethod
    def small_gather(x, ids, batch_dim=0):
        # use for loops...
        res = []
        B = x.shape[batch_dim]
        for b in range(B):
            res.append(x[b][ids[b]])
        res = torch.stack(res, dim=0)  # P x W x C
        return res

    def calc_MI(self, last):
        # nlogn
        NlogN = last['NlogN']  # W x P
        aloga = last['aloga']  # W x P
        blogb = last['blogb']  # W x P
        n = last['n']

        tN = (NlogN / n)
        ta = (-aloga / n)
        tb = (-blogb / n)
        term1 = (tN + ta + tb)
        term2 = n.log()
        scores = (term1 + term2)
        return scores

    def update_cache(self, last, idx, last_idx=None):
        # update nlogns
        for key in last.keys():
            if 'log' in key:
                last_idx = last_idx if last_idx is not None else idx
                self.cache[key] = last[key][last_idx]
        self.update_mats(idx)

    def calc_N_i(self, idx):
        # update N, a, b, n
        N_i = self.candidates['N'][idx]  # P x 2
        C = self.cache['N'].shape[-1]
        N_i = (self.one_hot(N_i[:, 0], C),
               self.one_hot(N_i[:, 1], C))
        N_i = torch.einsum('pa,pb->pab', N_i[0], N_i[1])  # P x C x C
        assert (N_i.sum([1, 2]) == 1).all(), 'error in getting one_hot representation of candidate'
        return N_i

    def update_mats(self, idx):
        N_i = self.calc_N_i(idx)
        self.cache['N'] += N_i
        self.cache['a'] += N_i.sum(1)
        self.cache['b'] += N_i.sum(2)
        self.cache['n'] += 1

    def add_samples(self, ids):
        for idx in ids:
            candidate = self.calc_N([idx], save_ids=False)
            last = self.get_last(candidate)
            self.update_cache(last, idx, last_idx=0)
