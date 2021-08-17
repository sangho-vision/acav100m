from slowfast.datasets.utils import (
    tensor_normalize,
    spatial_sampling as _spatial_sampling,
    pack_pathway_output
)


def preprocess(cfg, x):
    x = tensor_normalize(x, cfg.DATA.MEAN, cfg.DATA.STD)
    # T H W C -> C T H W.
    x = x.permute(3, 0, 1, 2)
    x = spatial_sampling(cfg, x)
    x = pack_pathway_output(cfg, x)
    if isinstance(x, list):
        if any(v is None for v in x):
            return None
    return x


def spatial_sampling(cfg, frames):
    min_scale, max_scale, crop_size = [cfg.DATA.TEST_CROP_SIZE] * 3

    frames = _spatial_sampling(
        frames,
        spatial_idx=0,
        min_scale=min_scale,
        max_scale=max_scale,
        crop_size=crop_size,
        random_horizontal_flip=cfg.DATA.RANDOM_FLIP,
        inverse_uniform_sampling=cfg.DATA.INV_UNIFORM_SAMPLE,
    )

    return frames
