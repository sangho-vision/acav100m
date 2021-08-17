import torch

from .efficient import EfficientMI


'''
pair-based scores
- Fowlkes–Mallows index
- Rand index
'''


def tensor_calc_combination(t, n=2):
    device = t.device
    dtype = t.dtype
    t = t.cpu().numpy()
    from scipy.special import comb
    t = comb(t, n, exact=False)
    return torch.from_numpy(t).round().to(dtype).to(device)


class FowlkesMallowsScore(EfficientMI):
    def init_cache(self):
        super().init_cache()
        self.pair_stats = ['TP', 'FP', 'FN', 'TN']
        P = len(self.combinations)
        self.cache = {**self.cache, **{name: torch.full(tuple([P]), self.eps) for name in self.pair_stats}}

    def add_samples(self, ids):
        super().add_samples(ids)  # W x P x C x C
        self.init_pair_stats()

    def init_pair_stats(self):
        N = self.cache['N']  # P x C x C
        a = self.cache['a']  # P x C
        b = self.cache['b']  # P x C
        n = self.cache['n']  # P

        S_ab = tensor_calc_combination(N, 2).sum(dim=[-1, -2])  # P
        S_a = tensor_calc_combination(a, 2).sum(dim=-1)  # P
        S_b = tensor_calc_combination(b, 2).sum(dim=-1)  # P
        n = tensor_calc_combination(n, 2)  # P
        res = self._calc_pair_stats(S_ab, S_a, S_b, n)

        for key in res.keys():
            self.cache[key] = res[key]

    def get_last(self):
        last = {key: self.cache[key].unsqueeze(0) * self.candidates[key]
                for key in self.candidates.keys()}
        return last

    def calc_pair_stats(self, last):
        N = last['N']  # W x P x C x C
        a = last['a']  # W x P x C
        b = last['b']  # W x P x C
        n = last['n']  # W x P

        S_ab = N.sum(dim=[-1, -2])  # W x P
        S_a = a.sum(dim=-1)
        S_b = b.sum(dim=-1)
        return self._calc_pair_stats(S_ab, S_a, S_b, n)

    def _calc_pair_stats(self, S_ab, S_a, S_b, n):
        S_aub = S_a + S_b - S_ab
        res = {
            'TP': S_ab,
            'FP': S_a - S_ab,
            'FN': S_b - S_ab,
            'TN': n - S_aub
        }
        return res

    def update_cache(self, last, idx):
        for key in self.candidates.keys():
            self.cache[key] += self.candidates[key][idx]
        for key in self.temp_pair_stats.keys():
            self.cache[key] += self.temp_pair_stats[key][idx]
        del self.temp_pair_stats

        self.pair_stats_sanity_check()

    def pair_stats_sanity_check(self):
        left_hand = sum([self.cache[key] for key in self.pair_stats])  # P
        n = self.cache['n']  # P
        right_hand = (n * (n - 1)) / 2  # P
        assert (left_hand == right_hand).all(), "pair stats count error"

    def _calc_score(self, *args, **kwargs):
        return self.calc_pair_score(*args, **kwargs)

    def calc_pair_score(self, last):
        pair_stats = self.calc_pair_stats(last)
        # W x P
        self.temp_pair_stats = pair_stats
        c = {p: self.cache[p].unsqueeze(0) + v for p, v in pair_stats.items()}
        return self._calc_pair_score(c)

    def _calc_pair_score(self, c):
        return self.calc_FM(c)

    def calc_FM(self, c):
        FM = ((c['TP'] / (c['TP'] + c['FP'])) * (c['TP'] / (c['TP'] + c['FN']))).sqrt()
        return FM


class RandScore(FowlkesMallowsScore):
    def _calc_pair_score(self, c):
        return self.calc_Rand(c)

    def calc_Rand(self, c):
        rand = (c['TP'] + c['TN']) / (c['TP'] + c['FP'] + c['FN'] + c['TN'])
        return rand


class AdjustedRandScore(EfficientMI):
    # TODO
    def _calc_score(self, *args, **kwargs):
        return self.calc_ARand(*args, **kwargs)

    def calc_ARand(self, last):
        N = last['N']  # W x P x C x C
        a = last['a']  # W x P x C
        b = last['b']
        n = last['n']  # W x P

        Nc = tensor_calc_combination(N, 2).sum(dim=[-1, -2])
        ac = tensor_calc_combination(a, 2).sum(dim=-1)
        bc = tensor_calc_combination(b, 2).sum(dim=-1)
        nc = tensor_calc_combination(n, 2)

        chance_term = (ac * bc) / nc
        numerator = Nc - chance_term
        denominator = 1 / 2 * (ac + bc) - chance_term
        return numerator / denominator
