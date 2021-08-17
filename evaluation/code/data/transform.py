import math
import numpy as np
import torch
import torchaudio


def random_short_side_scale_jitter(
    images, min_size, max_size,
):
    """
    Perform a spatial short scale jittering on the given images.
    Args:
        images (tensor): images to perform scale jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        min_size (int): the minimal size to scale the frames.
        max_size (int): the maximal size to scale the frames.
    Returns:
        (tensor): the scaled images with dimension of
            `num frames` x `channel` x `new height` x `new width`.
    """
    size = int(round(np.random.uniform(min_size, max_size)))

    height = images.shape[2]
    width = images.shape[3]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return images
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))

    return torch.nn.functional.interpolate(
        images,
        size=(new_height, new_width),
        mode="nearest",
        # align_corners=False,
    )


def random_crop(images, size):
    """
    Perform random spatial crop on the given images.
    Args:
        images (tensor): images to perform random crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): the size of height and width to crop on the image.
    Returns:
        cropped (tensor): cropped images with dimension of
            `num frames` x `channel` x `size` x `size`.
    """
    if images.shape[2] == size and images.shape[3] == size:
        return images
    height = images.shape[2]
    width = images.shape[3]
    y_offset = 0
    if height > size:
        y_offset = int(np.random.randint(0, height - size))
    x_offset = 0
    if width > size:
        x_offset = int(np.random.randint(0, width - size))
    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]

    return cropped


def horizontal_flip(prob, images):
    """
    Perform horizontal flip on the given images.
    Args:
        prob (float): probility to flip the images.
        images (tensor): images to perform horizontal flip, the dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): images with dimension of
            `num frames` x `channel` x `height` x `width`.
    """
    if np.random.uniform() < prob:
        images = images.flip((-1))

    return images


def uniform_crop(images, size, spatial_idx):
    """
    Perform uniform spatial sampling on the images.
    args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
    returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
    """
    assert spatial_idx in [0, 1, 2]
    height = images.shape[2]
    width = images.shape[3]

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]

    return cropped


def color_normalization(images, mean, stddev):
    """
    Perform color nomration on the given images.
    Args:
        images (tensor): images to perform color normalization. Dimension is
            `num frames` x `channel` x `height` x `width`.
        mean (list): mean values for normalization.
        stddev (list): standard deviations for normalization.

    Returns:
        out_images (tensor): the noramlized images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    assert len(mean) == images.shape[1], "channel mean not computed properly"
    assert (
        len(stddev) == images.shape[1]
    ), "channel stddev not computed properly"

    out_images = torch.zeros_like(images)
    for idx in range(len(mean)):
        out_images[:, idx] = (images[:, idx] - mean[idx]) / stddev[idx]

    return out_images


def mask_along_axis_consistent(
        specgrams,
        mask_param,
        mask_value,
        axis,
):
    """
    Apply a consistent mask along ``axis`` on a batch of specgrams.
    Mask will be applied from indices ``[v_0, v_0 + v)``, where
    ``v`` is sampled from ``uniform(0, mask_param)``, and ``v_0`` from ``uniform(0, max_v - v)``.
    Args:
        specgrams (Tensor): Real spectrograms (batch, channel, freq, time)
        mask_param (int): Number of columns to be masked will be uniformly sampled from [0, mask_param]
        mask_value (float): Value to assign to the masked columns
        axis (int): Axis to apply masking on (2 -> frequency, 3 -> time)
    Returns:
        Tensor: Masked spectrograms of dimensions (batch, channel, freq, time)
    """

    if axis != 2 and axis != 3:
        raise ValueError('Only Frequency and Time masking are supported')

    device = specgrams.device
    dtype = specgrams.dtype

    value = torch.rand(specgrams.shape[1], device=device, dtype=dtype) * mask_param
    min_value = torch.rand(specgrams.shape[1], device=device, dtype=dtype) * (specgrams.size(axis) - value)

    value, min_value = value.unsqueeze(0), min_value.unsqueeze(0)

    # Create broadcastable mask
    mask_start = min_value[..., None, None]
    mask_end = (min_value + value)[..., None, None]
    mask = torch.arange(0, specgrams.size(axis), device=device, dtype=dtype)

    # Per batch example masking
    specgrams = specgrams.transpose(axis, -1)
    specgrams.masked_fill_((mask >= mask_start) & (mask < mask_end), mask_value)
    specgrams = specgrams.transpose(axis, -1)

    return specgrams


def resample(waveform, orig_freq, new_freq, use_mono=True):
    """
    Resample the input waveform to ``new_freq``.
    args:
        waveform (tensor): waveform to perform resampling. The dimension is
            `channel` x `frequency` x `width`.
        `orig_freq` (int): original sampling rate of `waveform`.
        `new_freq` (int): target sampling rate of `waveform`.
        `use_mono` (bool): If True, first convert `waveform` to a monophonic signal.
    returns:
         (tensor): waveform with dimension of
            `channel` x `time`.
    """
    if waveform.size(0) != 1 and use_mono:
        waveform = waveform.mean(0, keepdim=True)

    if orig_freq != new_freq:
        waveform = torchaudio.transforms.Resample(
            orig_freq, new_freq,
        )(waveform)

    return waveform


def get_log_mel_spectrogram(
    waveform,
    audio_fps,
    frequency,
    time,
):
    """
    Convert the input waveform to log-mel-scaled spectrogram.
    args:
        waveform (tensor): input waveform. The dimension is
            `channel` x `time.`
        `audio_fps` (int): sampling rate of `waveform`.
        `frequency` (int): target frequecy dimension (number of mel bins).
        `time` (int): target time dimension.
    returns:
         (tensor): log-mel-scaled spectrogram with dimension of
            `channel` x `frequency` x `time`.
    """
    w = waveform.size(-1)
    n_fft = 2 * (math.floor(w / time) + 1)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        audio_fps, n_fft=n_fft, n_mels=frequency,
    )(waveform)
    log_mel_spectrogram = torch.log(1e-6 + mel_spectrogram)
    _nchannels, _frequency, _time = log_mel_spectrogram.size()
    assert _frequency == frequency, \
        f"frequency {_frequency} must be {frequency}"
    if _time != time:
        t = torch.zeros(
            _nchannels,
            frequency,
            time,
            dtype=log_mel_spectrogram.dtype,
        )
        min_time = min(time, _time)
        t[:, :, :min_time] = log_mel_spectrogram[:, :, :min_time]
        log_mel_spectrogram = t

    return log_mel_spectrogram
