import numpy as np
import torch
from sklearn import decomposition


def pca_features(features, dim=None):
    if dim is None:
        dim = min([m.shape[1] for m in features.values()])
    features = {k: pca_feature(v, dim) for k, v in features.items()}
    return features


def pca_feature(feature, dim=None):
    # features: K x D
    if torch.is_tensor(feature):
        feature = feature.numpy()
    assert isinstance(feature, np.ndarray), 'feature is nor a tensor or a numpy ndarray'
    pca = decomposition.PCA(n_components=dim)
    feature_pca = pca.fit_transform(feature)  # K x dim
    return feature_pca
