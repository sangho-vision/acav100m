import torch

from .mem_mi import EfficientMemMI
from .batch import EfficientBatchMI


class EfficientGpuMI(EfficientBatchMI, EfficientMemMI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.B = self.assignments.shape[0]
        self.k = 1
        self.keep_unselected = False

    def init_cache(self):
        EfficientBatchMI.init_cache(self)
        for key, val in self.cache.items():
            self.cache[key] = val.to(self.device)

    def init_candidates(self, candidates):
        EfficientBatchMI.init_candidates(self, candidates)
        EfficientMemMI.init_candidates(self, candidates)

    def calc_N_i(self, idx):
        return super().calc_N_i(idx, device=self.device)

    def get_last(self, candidates=None):
        if isinstance(candidates, dict):
            for k, v in candidates.items():
                if torch.is_tensor(v):
                    candidates[k] = v.to(self.device)
        elif torch.is_tensor(candidates):
            candidates = candidates.to(self.device)
        return super().get_last(candidates)

    def update_candidates(self, ids):
        idx = ids[0].item()  # singleton
        self.remove_idx_all(idx)

    def block_operate(self):
        batch_ranges = self.get_batch_ranges()

        block_scores = []
        block_samples = []
        for batch_range in batch_ranges:
            scores, samples = self.operate_block(batch_range)
            block_scores.append(scores)
            block_samples.append(samples)

        scores = torch.cat(block_scores, dim=0)
        samples = torch.cat(block_samples, dim=0)
        scores, ids = self.calc_ids(scores)
        candidate_ids = samples.to(self.device).index_select(0, ids).cpu()
        return scores, ids, None, candidate_ids

    def sample_batch(self, batch_range):
        if batch_range is None:
            batch_range = [0, self.B]
        if torch.is_tensor(batch_range):
            batch_ids = batch_range
            batch = self.candidate_ids.index_select(0, batch_ids)  # B
        else:
            batch = self.candidate_ids[batch_range[0]: batch_range[1]]  # B
        candidates = batch
        batch = self.calc_N(batch, save_ids=False)
        return {'N': batch}, candidates

    def get_batch_ranges(self):
        # chunk whole candidate set
        W = self.candidates['N'].shape[0]
        return super().get_batch_ranges(W)

    def update_cache(self, batch_update, ids):
        '''
        mem_mi_class = [c for c in self.mro() if 'MemMI' in c.__name__][0]
        return mem_mi_class.update_cache(*args, **kwargs)
        '''
        ids = ids.cpu().numpy().tolist()
        # ids = [self.candidate_ids[idx] for idx in ids]
        self.add_samples(ids)

    def add_samples(self, ids):
        EfficientMemMI.add_samples(self, ids)

    def calc_measure_batch(self):
        scores, ids, batch_update, candidate_ids = self.block_operate()
        self.update_cache(batch_update, ids)
        scores = scores.cpu()
        # use ids
        self.update_candidates(ids)
        return scores, candidate_ids, 1

    def calc_measure(self, celf=False):
        scores, ids, lookups = self.calc_measure_batch()
        return scores[0].item(), ids[0].item(), lookups

    def run(self, *args, **kwargs):
        return EfficientMemMI.run(self, *args, **kwargs)
