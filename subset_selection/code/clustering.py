from collections import defaultdict


class Clustering(object):
    def __init__(self, ncentroids, assignments):
        self.ncentroids = ncentroids
        self.ind2cen = assignments
        self.cen2ind = defaultdict(list)
        for ind, cen in enumerate(assignments):
            self.cen2ind[cen].append(ind)

    def get_cluster(self, cen):
        return self.cen2ind[cen]

    def get_assignment(self, ind):
        return self.ind2cen[ind]

    def get_num_agreed_pairs(self):
        num_agreed = 0
        for i in range(self.ncentroids):
            num_points = len(self.cen2ind[i])
            num_agreed += num_points * (num_points - 1) // 2
        return num_agreed
