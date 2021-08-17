"""Model construction functions."""

import torch
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video/audio model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg, model=None):
    """
    Builds the audio/video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in config.py.
        model (torch.nn.Module): Model module.
    """
    if model is None:
        # Construct the model
        name = cfg.MODEL.TASK
        model = MODEL_REGISTRY.get(name)(cfg)
    # Determine the GPU used by the current process
    cur_device = torch.cuda.current_device()
    # Transfer the model to the current GPU device
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        broadcast_buffers = False
        if cfg.MODEL.TASK in ["Contrast"] and cfg.CONTRAST.USE_GLOBAL_BATCH:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            broadcast_buffers = True
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[cur_device],
            output_device=cur_device,
            broadcast_buffers=broadcast_buffers,
        )
    return model
