from typing import List

import scipy
import torch

from clustering import Clustering


'''
Not Implemented
hungarian = scipy.optimize.linear_sum_assignment
row_ind, col_ind = hungarian(cost_matrix)
'''

class EfficientDMC:
    """ this implementation requires the users to use the same ncentroids for all clusterings """
    """ we need clustering algorithm with multiple assignments per case """
    def __init__(self, clusterings: List[Clustering], measure_type='mutual_info'):
        self.clusterings = clusterings
        self.eps = 1e-20

    def init_cache(self):
        P = len(self.combinations)
        C = self.clusterings[0].ncentroids
        N = torch.full((P, C), self.eps)
        n = N.sum(dim=[-1, -2])
        self.cache = {'N': N, 'n': n}
