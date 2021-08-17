import numpy as np


def get_weights(keys, pairing, weight_type=None):
    # e.g. [(0,0), (2,3), ...]
    if weight_type is None:
        return pairing
    n_layer = (np.array(pairing).max() + 1) // 2
    weights = _get_weights(n_layer, weight_type)  # n_layer
    weights = np.concatenate([weights, weights])
    print('weights', weights)
    pairing_weights = [weights[v[0]] * weights[v[1]] for v in pairing]
    return {'pairing': pairing, 'weights': pairing_weights}


def _get_weights(n_layer, weight_type=None):
    eps = 1e-10
    weight_type = weight_type.split('_')
    func_name = weight_type[0]
    if func_name == 'onehot':
        weights = np.array([float(0)] * n_layer)
        coeff = 0
        if len(weight_type) == 2:
            coeff = weight_type[1]
            weights[int(coeff)] = 1
            return weights

    coeff = 1.0
    if len(weight_type) == 2:
        coeff = weight_type[1]
        coeff = float(coeff)

    # linear, log, exp...
    func = {
        'linear': lambda x: x,
        'log': lambda x: np.log(x),
        'exp': lambda x: np.exp(x),
    }[func_name]

    # assume odd n_layer
    # keep mean=1
    mean = (1 + n_layer) / 2
    x = np.arange(float(n_layer)) - mean
    weights = x * coeff + 1
    minv = weights.min()
    weights = weights - minv + 2  # for log stabilization
    weights = func(weights)
    # normalize
    weights = weights / np.median(weights)
    return weights  # n_layer lengthed np vector
