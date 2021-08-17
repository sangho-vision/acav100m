import math
import time
import json
from typing import List, Tuple
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from .efficient import EfficientMI


class EfficientBatchMI(EfficientMI):
    def __init__(self, *args, batch_size=1, selection_size=1,
                 device='cpu', keep_unselected=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.B = batch_size
        self.k = selection_size
        self.device = device
        self.keep_unselected = keep_unselected

    def init_candidates(self, candidates):
        self.candidate_ids = torch.LongTensor(candidates)
        # use self.assignments

    def init_cache(self):
        super().init_cache()
        for key, val in self.cache.items():
            self.cache[key] = val.to(self.device)

    def shuffle_candidate_ids(self):
        L = self.candidate_ids.shape[0]
        idx = torch.randperm(L).to(self.candidate_ids.device)
        self.candidate_ids = self.candidate_ids.index_select(0, idx)

    def sample_batch(self, batch_range):
        if batch_range is None:
            batch_range = [0, self.B]
        if torch.is_tensor(batch_range):
            batch_ids = batch_range
            batch = self.candidate_ids.index_select(0, batch_ids)  # B
        else:
            batch = self.candidate_ids[batch_range[0]: batch_range[1]]  # B
        assignments = self.get_assignments(batch)
        N = self.get_N(assignments)
        a = N.sum(2)
        b = N.sum(3)
        n = b.sum(-1)
        return {'N': N, 'a': a, 'b': b, 'n': n}, batch

    def get_batch_ranges(self, B=None):
        if B is None:
            B = self.B
        single_range = [[0, B]]
        if self.device == 'cpu':
            return single_range

        B = self.B
        P = len(self.combinations)
        C = self.ncentroids
        float_size = 32
        expansion = 0.9

        total_mem = torch.cuda.get_device_properties(self.device).total_memory
        # DEBUG
        # total_mem = 20 * 45 * 10 * 10 * float_size * 0.3
        mem_consumption = B * P * C * C * float_size

        if mem_consumption * expansion < total_mem:
            return single_range
        else:
            num_chunks = math.ceil(mem_consumption * expansion / total_mem)
            chunk_size = self.B // num_chunks
            all_ranges = []
            start = 0
            end = start + chunk_size
            for i in range(num_chunks):
                all_ranges.append((start, end))
                start += chunk_size
                end += chunk_size
                start = min(start, self.B)
                end = min(end, self.B)
                if start == end:
                    break
            return all_ranges

    def get_range_subset(self, ids, brange):
        mask = (ids >= brange[0]) & (ids < brange[1])
        return ids[mask]

    def block_operate(self):
        self.shuffle_candidate_ids()

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

        if hasattr(self, 'gold_standard'):
            # calc batch ratio
            total_num = samples.shape[0]
            positives = len(set(self.gold_standard) & set(samples.cpu().numpy()))
            self.batch_ratios.append((positives / total_num))

        block_batch_update = None
        for batch_range in batch_ranges:
            batch_ids = self.get_range_subset(ids, batch_range)
            batch, _ = self.sample_batch(batch_ids.to('cpu'))
            batch = {k: v.to(self.device).sum(dim=0) for k, v in batch.items()}
            if block_batch_update is None:
                block_batch_update = batch
            else:
                for k, v in batch.items():
                    block_batch_update[k] += v

        candidate_ids = samples.to(self.device).index_select(0, ids).cpu()
        return scores, ids, block_batch_update, candidate_ids

    def operate_block(self, batch_range=None):
        if batch_range is None:
            batch_range = [0, self.B]
        batch, samples = self.sample_batch(batch_range)
        last = self.get_last(batch)
        scores = self._calc_score(last)
        del batch
        return scores, samples

    def calc_measure_batch(self):
        scores, ids, batch_update, candidate_ids = self.block_operate()
        self.update_cache(batch_update, ids)
        scores = scores.cpu()
        self.update_candidates(candidate_ids)
        return scores, candidate_ids, 1

    def calc_score(self, *args, **kwargs):
        scores = self._calc_score(*args, **kwargs)
        return self.calc_ids(scores)

    def calc_ids(self, scores):
        scores = scores.mean(dim=-1)  # B
        scores, ids = scores.topk(k=self.k, dim=0)
        return scores, ids

    def update_cache(self, batch_update, ids):
        for key in batch_update.keys():
            self.cache[key] += batch_update[key]

    def update_candidates(self, candidate_ids):
        batch = self.candidate_ids[:self.B]
        self.candidate_ids = self.candidate_ids[self.B:]  # remove current batch
        if self.keep_unselected:
            unselected = self.get_unselected(batch, candidate_ids)
            unselected = unselected.to(self.candidate_ids.device)
            assert unselected.shape[0] + candidate_ids.shape[0] == batch.shape[0], 'wrong unselected_size: unselected {} + {} != {}'.format(
                unselected.shape[0], candidate_ids.shape[0], batch.shape[0])
            self.candidate_ids = torch.cat((self.candidate_ids, unselected), dim=0)

    def get_unselected(self, orig, selected):
        if torch.is_tensor(selected):
            orig = orig.to(selected.device)
        comb = torch.cat((orig, selected), dim=0)
        uniques, counts = comb.unique(return_counts=True)
        diff = uniques[counts == 1]
        return diff

    def modify_k(self, subset_size):
        dataset_size = self.assignments.shape[0]
        D = dataset_size
        S = subset_size
        K = self.k
        B = self.B

        term = B * S / D

        if K < term and not self.keep_unselected:
            print("k={} is too small to get {} samples from {} datapoints with batch_size {}".format(
                K, S, D, B))
            K = math.ceil(term)
            print("resizing k to {}".format(K))

        return K

    def add_samples(self, candidate_ids):
        to_add = self._add_samples(candidate_ids)
        for idx in candidate_ids:
            self.remove_idx_all(idx)
        for key in to_add.keys():
            self.cache[key] += to_add[key].to(self.device)

    def run(self, subset_size, start_indices, intermediate_target=None, celf_ratio=0):
        # do not use celf_ratio
        S = start_indices
        GAIN = []
        LOOKUPS = []
        timelapse = []

        print('using {} blocks per iter for gpu computation'.format(len(self.get_batch_ranges())))

        if intermediate_target is not None:
            self.gold_standard = np.array(intermediate_target)  # only for calculating batch ratio stats
            self.batch_ratios: List[Tuple[float, float]] = []  # store

        self.k = self.modify_k(subset_size)

        dataset_size = self.candidate_ids.shape[0]
        assert dataset_size == self.candidate_ids.unique().shape[0], "duplicates in initial candidate_ids"

        self.add_samples(start_indices)

        greedy_start_time = time.time()
        start_time = time.time()
        # start from empty index
        iters = list(range(len(start_indices), subset_size - 1))
        niters = len(iters)
        pbar = tqdm(iters, total=niters, desc='iter')
        while len(S) < subset_size:
            start_time = time.time()
            scores, ids, lookup = self.calc_measure_batch()
            scores = scores.numpy().tolist()
            ids = ids.numpy().tolist()
            timelapse.append(time.time() - start_time)
            S += ids
            GAIN += scores
            LOOKUPS.append(lookup)

            if self.keep_unselected:
                assert self.candidate_ids.shape[0] + len(S) == dataset_size, "dataset size mismatch: {} + {} != {}".format(
                    self.candidate_ids.shape[0], len(S), dataset_size)
                assert self.candidate_ids.unique().shape[0] + len(S) == dataset_size, "dataset unique key size mismatch: {} + {} != {}".format(
                    self.candidate_ids.unique().shape[0], len(S), dataset_size)

            pbar.update(len(ids))

            if intermediate_target is not None:
                precision = len(set(intermediate_target) & set(S)) / len(set(S))
                pbar.set_description("(LEN: {}, MEASURE: {}, PRECISION: {})".format(
                    len(S), np.array(scores).mean(), precision))
            else:
                pbar.set_description("(LEN: {}, MEASURE: {})".format(len(S), np.array(scores).mean()))

        S = S[:subset_size]  # cut rest
        if hasattr(self, 'gold_standard'):
            batch_ratios = self.batch_ratios[:subset_size]  # cut rest
            with open('batch_ratios.json', 'w') as f:
                json.dump(batch_ratios, f, indent=4)

        tqdm.write("Time Consumed: {} seconds".format(time.time() - greedy_start_time))
        return (S, GAIN, timelapse, LOOKUPS)
