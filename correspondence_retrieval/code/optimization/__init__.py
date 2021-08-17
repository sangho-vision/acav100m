from .celf import celf
from .greedy import greedy
from .efficient import efficient_greedy


def optimize(*args, algorithm='celf', **kwargs):
    alg_dict = {
        'celf': celf,
        'greedy': greedy,
        'efficient_greedy': efficient_greedy
    }
    algorithm = algorithm.lower()
    assert algorithm in alg_dict, f"invalid optimization algorithm type: {algorithm}"
    return alg_dict[algorithm](*args, **kwargs)
