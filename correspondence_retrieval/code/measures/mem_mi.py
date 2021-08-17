import torch

from .efficient import EfficientMI


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

    def check_cache(self, N=None, name='N'):
        if N is None:
            N = self.cache['N']
        x = torch.arange(N.nelement()).reshape(*N.shape).float().to(N.device)
        print(name, (N * x).var())

    def get_last(self, candidates=None):
        if candidates is None:
            candidates = self.candidates

        N = self.get_last_N(self.cache['N'], candidates['N'])  # W x P
        a = self.get_last_ab(self.cache['a'], candidates['N'], dim=1)  # W x P
        b = self.get_last_ab(self.cache['b'], candidates['N'], dim=0)  # W x P
        last = {}
        last['NlogN'] = self.get_nlogn(self.cache['NlogN'].unsqueeze(0), N)
        last['aloga'] = self.get_nlogn(self.cache['aloga'].unsqueeze(0), a)
        last['blogb'] = self.get_nlogn(self.cache['blogb'].unsqueeze(0), b)
        last['n'] = (self.cache['n'] + 1).unsqueeze(0)

        return last

    @staticmethod
    def nlogn(x):
        return x * x.log()

    def get_nlogn(self, prev, num):
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

    def update_cache(self, *args, **kwargs):
        return self._update_cache(*args, **kwargs)

    def _update_cache(self, last, idx, last_idx=None):
        # update nlogns
        for key in last.keys():
            if 'log' in key:
                last_idx = last_idx if last_idx is not None else idx
                self.cache[key] = last[key][last_idx]
        self.update_mats(idx)

    def calc_N_i(self, idx, device='cpu'):
        # update N, a, b, n
        N_i = self.candidates['N'][idx]  # P x 2
        C = self.cache['N'].shape[-1]
        N_i = N_i.to(device)
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
            self.update_sample(idx)

    def update_sample(self, idx):
        # candidate = self.calc_N([idx], save_ids=False)
        candidate = self.candidates['N'][idx].unsqueeze(0)
        last = self.get_last({'N': candidate})
        self._update_cache(last, idx, last_idx=0)
