from collections import defaultdict

from tqdm import tqdm

import torch
import numpy as np
from scipy.cluster.vq import whiten

from sgd_clustering import sgd_kmeans_clustering
from pca import pca_features


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


def shard_clustering(clustering, shard_ids):
    clusterings = []
    ncentroids = clustering.ncentroids
    for i in range(len(shard_ids)):
        assignments = np.array([clustering.ind2cen[p] for p in shard_ids[i]])
        sharded_clustering = Clustering(ncentroids, assignments)
        clusterings.append(sharded_clustering)
    return clusterings


def run_pipe(views_features, pipe, name=None):
    if name is not None:
        print("Running {}".format(name))
    res = {}
    for view, features in tqdm(views_features.items()):
        res[view] = pipe(features)
    return res


def cat_and_whiten(x):
    if x[0].dim() == 2:
        x = torch.cat(x, dim=0)
    elif x[0].dim() == 1:
        x = torch.stack(x, dim=0)
    return whiten(x.numpy())


def run_clusterings(args, all_features, ncentroids=10, kmeans_niters=None, clustering_func_type='scipy'):
    all_features = run_pipe(all_features, cat_and_whiten, 'whiten')
    if 'pca' in args.measure:
        clustering_func_type = 'pca'
    clustering_func_dict = {
        'scipy': scipy_clusterings,
        'faiss': faiss_clusterings,
        'sgd_kmeans': sgd_kmeans_clusterings,
        'pca': pca_features # not a clustering, but we put it here for encapsulation
    }
    if not isinstance(ncentroids, list):
        ncentroids = [ncentroids]
    assert clustering_func_type in clustering_func_dict, "invalid clustering_func_type: {}".format(clustering_func_type)
    clustering_func = clustering_func_dict[clustering_func_type]
    all_clusterings = {}
    if clustering_func_type == 'pca':
        all_clusterings = clustering_func(all_features, args.pca_dim)
    else:
        for ncentroid in ncentroids:
            centroid_assignments = clustering_func(args, all_features, ncentroid, kmeans_niters)
            centroid_clusterings = run_pipe(centroid_assignments, lambda x: Clustering(ncentroid, x), 'clustering')
            for k, clustering in centroid_clusterings.items():
                all_clusterings[f"{k}_{ncentroid}"] = clustering
    return all_clusterings


def scipy_clusterings(args, all_features, ncentroids, kmeans_niters):
    from scipy.cluster.vq import vq, kmeans
    if kmeans_niters is None:
        kmeans_niters = 20
    all_codebooks = run_pipe(all_features, lambda x: kmeans(x, ncentroids, kmeans_niters)[0], 'kmeans')
    vq_args = {view: [all_features[view], all_codebooks[view]] for view in all_features.keys()}
    all_assignments = run_pipe(vq_args, lambda x: vq(x[0], x[1])[0], 'vq')
    return all_assignments


def faiss_clustering(features, ncentroids, kmeans_niters):
    import faiss
    d = features.shape[1]
    if kmeans_niters is None:
        kmeans_niters = 20
    kmeans = faiss.Kmeans(d, ncentroids, niter=kmeans_niters, verbose=False)
    kmeans.train(features)
    distances, assignments = kmeans.index.search(features, 1)
    return assignments.squeeze(axis=-1)


def faiss_clusterings(args, all_features, ncentroids, kmeans_niters):
    all_assignments = run_pipe(all_features, lambda x: faiss_clustering(x, ncentroids, kmeans_niters), 'kmeans')
    return all_assignments


def sgd_kmeans_clusterings(args, all_features, ncentroids, kmeans_niters):
    all_assignments = run_pipe(all_features, lambda x: sgd_kmeans_clustering(args, x, ncentroids, kmeans_niters), 'sgd_kmeans')
    return all_assignments
