from typing import List

from .custom_measure import CustomMeasure
from .mutual_information import MutualInformation, MEASURES
from .efficient import (
    EfficientMI,
    EfficientAMI,
    EfficientNMI,
    ConstantMeasure
)
from .efficient_pair import (
    FowlkesMallowsScore,
    RandScore,
    AdjustedRandScore
)
from .mem_mi import EfficientMemMI
from .batch import EfficientBatchMI
from .mi_gpu import EfficientGpuMI
from .pca import PCAOptim, DISTANCES
from .metric import MetricLearning


EFFICIENT_MEASURES = {
    'efficient_mi': EfficientMI,
    'efficient_ami': EfficientAMI,
    'efficient_nmi': EfficientNMI,
    'efficient_mem_mi': EfficientMemMI,
    'efficient_batch_mi': EfficientBatchMI,
    'efficient_gpu_mi': EfficientGpuMI,
    'efficient_fm': FowlkesMallowsScore,
    'efficient_rand': RandScore,
    'efficient_arand': AdjustedRandScore,
    'efficient_constant': ConstantMeasure,
    'pca': PCAOptim,
    'pca_ip': PCAOptim,
    'pca_cs': PCAOptim,
    'pca_l1': PCAOptim,
    'pca_l2': PCAOptim,
    'contrastive': MetricLearning
}


def get_measure(args, clusterings, measure='custom'):
    measure_dict = {
        'custom': CustomMeasure,
        # 'mutual_information': MutualInformation,
        **EFFICIENT_MEASURES,
        **{key: MutualInformation for key in MEASURES.keys()}
    }
    measure = measure.lower()
    assert measure in measure_dict, f"invalid optimization measure type: {measure}"
    if measure in ['efficient_batch_mi', 'efficient_gpu_mi']:
        return measure_dict[measure](clusterings,
                                     batch_size=args.batch_size,
                                     selection_size=args.selection_size,
                                     device=args.device,
                                     keep_unselected=args.keep_unselected)
    elif measure_dict[measure] == MutualInformation:
        return measure_dict[measure](clusterings, measure_type=MEASURES[measure])
    elif measure_dict[measure] == MetricLearning:
        return measure_dict[measure](args, clusterings)
    elif measure_dict[measure] == PCAOptim:
        return measure_dict[measure](clusterings, measure_type=DISTANCES[measure])
    else:
        return measure_dict[measure](clusterings)
