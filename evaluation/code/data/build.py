from fvcore.common.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""


def build_dataset(dataset_name, cfg, split):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the dataset to be constructed.
        cfg (CfgNode): configs. Details can be found in config.py.
        split (str): the split of the data loader. Options include `pretrain`,
            `train`, `val` and `test`.
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    name = dataset_name
    return DATASET_REGISTRY.get(name)(cfg, split)
