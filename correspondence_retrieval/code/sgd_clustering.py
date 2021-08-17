import math
from pathlib import Path

import torch
from numpy import newaxis
import torch_scatter

from utils import load_pickle, dump_pickle
from common import get_cache_path


def sgd_kmeans_clustering(args, features, ncentroids, kmeans_niters, sgd_batch_size=64,
                          verbose=False):
    out_path = Path(args.out_path)
    path = get_cache_path(args, out_path)
    '''
    if path.is_file():
        clustering = load_pickle(path)
    else:
        clustering = _sgd_kmeans_clustering(features, ncentroids, kmeans_niters, sgd_batch_size=sgd_batch_size,
                                            verbose=verbose)
        dump_pickle(clustering, path)
        '''
    clustering = _sgd_kmeans_clustering(features, ncentroids, kmeans_niters, sgd_batch_size=sgd_batch_size,
                                        verbose=verbose)
    return clustering


def _sgd_kmeans_clustering(features, ncentroids, kmeans_niters, sgd_batch_size=64,
                          verbose=False):
    if kmeans_niters is None:
        kmeans_niters = 20
    n = features.shape[0]
    d = features.shape[1]
    km = KMeans(d, ncentroids)
    is_gpu = torch.cuda.device_count() > 0
    if is_gpu:
        km.to('cuda')
    if verbose:
        print(f"training for {kmeans_niters} epochs")
    n_iters = math.ceil(n / sgd_batch_size)
    for epoch in range(kmeans_niters):
        km.lr = 0.1 ** (2 + epoch // 5)
        # print("epoch", epoch, "lr", km.lr)
        for i in range(n_iters):
            batch = features[i * sgd_batch_size: (i + 1) * sgd_batch_size]
            batch = torch.from_numpy(batch)
            if is_gpu:
                batch = batch.to('cuda')
            km.add(batch)
    if verbose:
        print(f"assigning clusters")
    assignments = []
    for i in range(n_iters):
        batch = features[i * sgd_batch_size: (i + 1) * sgd_batch_size]
        batch = torch.from_numpy(batch)
        if is_gpu:
            batch = batch.to('cuda')
        assignment = km.calc_best(batch)
        assignments.append(assignment.cpu())
    assignments = torch.cat(assignments, dim=0).numpy()
    return assignments


class KMeans:
    """KMeans by Gradient Descent.

    This uses Torch and can run on GPU. For initialization, it averages `initial_rounds` random assignments of samples to centers.
    In order to keep the number of unused cluster centers down, it uses a simple heuristic to kick cluster centers with low usage
    back into the search by discounting their distance computation (`reinit`). It's generally a good idea to decrease the learning
    rate proportional to N**-0.5 after N training samples for convergence.
    """
    def __init__(self, d, k, lr=1e-2, initial_rounds=10, reinit=(.7, 5.0)):
        self.centers = torch.rand(k, d) * 1e-5
        self.counts = torch.zeros(k, dtype=int)
        self.count = 0
        self.lr = lr
        self.initial_rounds = initial_rounds
        self.reinit = reinit
        self.fallback = 0
        self.sequential = False

    def to(self, device):
        self.centers = self.centers.to(device)
        self.counts = self.counts.to(device)

    def calc_best(self, batch):
        batch = batch.to(self.centers.device)
        k, d = self.centers.shape
        b = len(batch)
        if self.count < self.initial_rounds * k:
            distances = torch.rand(k, b, device=self.centers.device)
        else:
            with torch.no_grad():
                # compute distances by (x - y)**2 = -2*x*y + x**2 + y ** 2
                distances = - 2 * torch.matmul(self.centers, batch.T)
                distances += (torch.norm(batch, dim=1)**2)[newaxis, :]
                distances += (torch.norm(self.centers, dim=1)**2)[:, newaxis]
                # simple reinitialization logic for underused cluster centers
                p, r = self.reinit
                distances[self.counts < (self.count / k) ** p, :] /= r
        best = distances.argmin(axis=0)
        return best

    def add(self, batch):
        k, d = self.centers.shape
        b = len(batch)
        best = self.calc_best(batch)
        lr = self.lr(self.count) if callable(self.lr) else self.lr
        if self.sequential:
            # slow sequential update
            for i, j in enumerate(best):
                self.centers[j] *= 1 - lr
                self.centers[j] += lr * batch[i]
                self.counts[j] += 1
        else:
            # fast parallel update
            counts = torch_scatter.scatter_add(src=torch.ones(b, device=self.counts.device, dtype=torch.long), index=best, dim_size=k)
            counts = counts.to(self.centers.device)
            if counts.max().item() * lr >= 1.0:
                # for parallel updates, the lr is easily too high; we use this as a fallback
                lr = 0.5 / counts.max().item()
                self.fallback += 1
            self.counts += counts
            self.centers *= (1. - counts * lr)[:, newaxis]
            torch_scatter.scatter_add(src=batch*lr, index=best, out=self.centers, dim=0)
        self.count += b
