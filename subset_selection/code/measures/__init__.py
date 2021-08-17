from .mi import EfficientMI, EfficientAMI, EfficientMemMI
from .batch import EfficientBatchMI


def get_measure(measure_name):
    dt = {
        'mi': EfficientMI,
        'ami': EfficientAMI,
        'mem_mi': EfficientMemMI,
        'batch_mi': EfficientBatchMI
    }
    measure_name = measure_name.lower()
    assert measure_name in dt, "no measure named {}".format(measure_name)
    return dt[measure_name]
