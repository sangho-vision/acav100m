# from itertools import product
import fire
import numpy as np

from run import _run, get_stats
from compare_shards import run_exp


class Cli:
    def run(self, verbose=True, **kwargs):
        run_exp(verbose=verbose, **kwargs)

def format_stats(li):
    li = list(zip(*sorted([v.items() for v in li])))
    dt = {row[0][0]: np.array([v[1] for v in row]) for row in li}
    return dt


def get_means(dt):
    dt = {k: v.mean() for k, v in dt.items()}
    return dt


if __name__ == '__main__':
    fire.Fire(Cli)
