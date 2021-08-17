import torch
import torchaudio

import data.transform as transform


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


def apply_visual_transform(
    cfg,
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
):
    """
    Apply visual data transformations on the given video frames.
    Data transformations include `resize_crop`, `flip` and `color_normalize`.
    Args:
        cfg (CfgNode): configs.
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `channel` x `height` x `width`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        for _transform in cfg.DATA.TRANSFORMATIONS:
            if _transform == "resize_crop":
                frames = transform.random_short_side_scale_jitter(
                    images=frames,
                    min_size=min_scale,
                    max_size=max_scale,
                )
                frames = transform.random_crop(frames, crop_size)
            elif _transform == "flip":
                frames = transform.horizontal_flip(0.5, frames)
            elif _transform == "color_normalize":
                frames = transform.color_normalization(
                    frames,
                    cfg.DATA.MEAN,
                    cfg.DATA.STD,
                )
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames = transform.random_short_side_scale_jitter(
            frames, min_scale, max_scale,
        )
        frames = transform.uniform_crop(frames, crop_size, spatial_idx)
        if "color_normalize" in cfg.DATA.TRANSFORMATIONS:
            frames = transform.color_normalization(
                frames, cfg.DATA.MEAN, cfg.DATA.STD,
            )
    return frames


def apply_audio_transform(
    log_mel_spectrogram,
    frequency_mask_rate,
    time_mask_rate,
):
    """
    Apply audio data transformations on the given video frames.
    Data transformations include `resize_crop`, `flip` and `color_normalize`.
    Args:
        log_mel_spectrogram (tensor): log-mel-scaled spectrogram. The
            dimension is `channel` x `frequency` x `time`.
        frequency_mask_rate (float): maksing dropout rate in frequency dimension.
        time_mask_rate (float): masking dropout rate in time dimension.
    Returns:
        (tensor): log-mel-scaled spectrogram After applying transformations.
    """
    dim = log_mel_spectrogram.dim()
    assert dim == 3 or dim == 4, \
        f"# of dimension of log_mel_spectrogram ({dim}) must be 3 or 4"
    if log_mel_spectrogram.dim() == 3:
        frequency_mask_param = int(
            log_mel_spectrogram.size(1)
            * frequency_mask_rate
        )
        time_mask_param = int(
            log_mel_spectrogram.size(2)
            * time_mask_rate
        )
        log_mel_spectrogram = torchaudio.functional.mask_along_axis(
            log_mel_spectrogram,
            frequency_mask_param,
            0.0,
            1,
        )
        log_mel_spectrogram = torchaudio.functional.mask_along_axis(
            log_mel_spectrogram,
            time_mask_param,
            0.0,
            2,
        )
    else:
        frequency_mask_param = int(
            log_mel_spectrogram.size(2)
            * frequency_mask_rate
        )
        time_mask_param = int(
            log_mel_spectrogram.size(3)
            * time_mask_rate
        )
        log_mel_spectrogram = transform.mask_along_axis_consistent(
            log_mel_spectrogram,
            frequency_mask_param,
            0.0,
            2,
        )
        log_mel_spectrogram = transform.mask_along_axis_consistent(
            log_mel_spectrogram,
            time_mask_param,
            0.0,
            3,
        )

    return log_mel_spectrogram
