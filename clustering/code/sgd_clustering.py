import math

import torch
from numpy import newaxis
import torch_scatter

import mps.distributed as du


class KMeans:
    """KMeans by Gradient Descent.

    This uses Torch and can run on GPU. For initialization, it averages `initial_rounds` random assignments of samples to centers.
    In order to keep the number of unused cluster centers down, it uses a simple heuristic to kick cluster centers with low usage
    back into the search by discounting their distance computation (`reinit`). It's generally a good idea to decrease the learning
    rate proportional to N**-0.5 after N training samples for convergence.
    """
    def __init__(self, args=None, d=None, k=None, lr=1e-2,
                 initial_rounds=10, reinit=(.7, 5.0), saved_dt=None):
        if saved_dt is not None:
            self.load_from_saves(saved_dt)
        else:
            self.args = args
            self.centers = torch.rand(k, d) * 1e-5
            # self.counts = torch.zeros(k, dtype=int)
            self.counts = torch.zeros(k)
            self.count = 0
            self.lr = lr
            self.initial_rounds = initial_rounds
            self.reinit = reinit
            self.fallback = 0
            self.sequential = False

    def get_attrs(self):
        dt = {
            'args': self.args,
            'count': self.count,
            'lr': self.lr,
            'initial_rounds': self.initial_rounds,
            'reinit': self.reinit,
            'fallback': self.fallback,
            'sequential': self.sequential,
            'centers': self.centers.cpu().numpy(),
            'counts': self.counts.cpu().numpy(),
        }
        return dt

    def load_from_saves(self, dt):
        for k, v in dt.items():
            setattr(self, k, v)
        self.centers = torch.from_numpy(self.centers)
        self.counts = torch.from_numpy(self.counts)

    @classmethod
    def load(cls, dt):
        obj = cls(saved_dt=dt)
        return obj

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
        distances, best = distances.min(axis=0)
        return best, distances.mean().item()

    @property
    def is_distributed(self):
        return (
            self.args.computation.device == 'cuda' and
            self.args.computation.num_gpus > 1
        )

    def initialize(self):
        if self.is_distributed:
            self.centers, self.counts = du.all_reduce(
                [self.centers, self.counts],
            )

    def add(self, batch):
        k, d = self.centers.shape
        if self.is_distributed:
            [gbatch] = du.all_gather([batch])
        else:
            gbatch = batch
        lr = self.lr(self.count) if callable(self.lr) else self.lr
        b = len(gbatch)
        local_b = len(batch)
        if self.sequential:
            best, distances = self.calc_best(gbatch)
            # slow sequential update
            for i, j in enumerate(best):
                self.centers[j] *= 1 - lr
                self.centers[j] += lr * gbatch[i]
                self.counts[j] += 1
        else:
            best, distances = self.calc_best(batch)
            # fast parallel update
            counts = torch_scatter.scatter_add(src=torch.ones(local_b, device=self.counts.device, dtype=torch.float), index=best, dim_size=k).cuda()
            if self.is_distributed:
                [counts] = du.all_reduce([counts], average=False)
            if counts.max().item() * lr >= 1.0:
                # for parallel updates, the lr is easily too high; we use this as a fallback
                lr = 0.5 / counts.max().item()
                self.fallback += 1
            self.counts += counts
            self.centers *= (1. - counts * lr)[:, newaxis]
            deltas = torch.zeros_like(self.centers)
            torch_scatter.scatter_add(src=batch*lr, index=best, out=deltas, dim=0)
            # deltas = torch_scatter.scatter(src=batch*lr, index=best, dim=0)
            if self.is_distributed:
                [deltas] = du.all_reduce([deltas], average=False)
            self.centers = self.centers + deltas
        self.count += b
        return distances
