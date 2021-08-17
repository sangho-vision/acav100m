import torch
import numpy as np
import resampy

from .mel_features import log_mel_spectrogram


"""
The only difference of this code from the original repository is that
it ensures the outputs contain at least a single frame
"""


def _preprocess(data, sample_rate):
    # Architectural constants.
    NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
    NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
    EMBEDDING_SIZE = 128  # Size of embedding layer.

    # Hyperparameters used in feature and example generation.
    SAMPLE_RATE = 16000
    STFT_WINDOW_LENGTH_SECONDS = 0.025
    STFT_HOP_LENGTH_SECONDS = 0.010
    NUM_MEL_BINS = NUM_BANDS
    MEL_MIN_HZ = 125
    MEL_MAX_HZ = 7500
    LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
    EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
    EXAMPLE_HOP_SECONDS = 0.96  # with zero overlap.

    # Parameters used for embedding postprocessing.
    PCA_EIGEN_VECTORS_NAME = 'pca_eigen_vectors'
    PCA_MEANS_NAME = 'pca_means'
    QUANTIZE_MIN_VAL = -2.0
    QUANTIZE_MAX_VAL = +2.0

    # Hyperparameters used in training.
    INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.
    LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer.
    ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.

    # Names of ops, tensors, and features.
    INPUT_OP_NAME = 'vggish/input_features'
    INPUT_TENSOR_NAME = INPUT_OP_NAME + ':0'
    OUTPUT_OP_NAME = 'vggish/embedding'
    OUTPUT_TENSOR_NAME = OUTPUT_OP_NAME + ':0'
    AUDIO_EMBEDDING_FEATURE_NAME = 'audio_embedding'

    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    resampled = data
    # Resample to the rate assumed by VGGish.
    if sample_rate != SAMPLE_RATE:
        resampled = resampy.resample(resampled, sample_rate, SAMPLE_RATE)

    def get_log_mel(x):
        return log_mel_spectrogram(
        x,
        audio_sample_rate=SAMPLE_RATE,
        log_offset=LOG_OFFSET,
        window_length_secs=STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=NUM_MEL_BINS,
        lower_edge_hertz=MEL_MIN_HZ,
        upper_edge_hertz=MEL_MAX_HZ)

    log_mel = get_log_mel(resampled)
    # Frame features into examples.
    features_sample_rate = 1.0 / STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(
        EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(
        EXAMPLE_HOP_SECONDS * features_sample_rate))

    num_samples = log_mel.shape[0]

    num_frames = int(np.floor((num_samples - example_window_length) / example_hop_length))
    num_frames = 1 + num_frames

    shape = (num_frames, example_window_length) + log_mel.shape[1:]
    strides = (log_mel.strides[0] * example_hop_length,) + log_mel.strides
    log_mel_examples = np.lib.stride_tricks.as_strided(log_mel, shape=shape, strides=strides)

    log_mel_examples_tensor = torch.tensor(
        log_mel_examples, requires_grad=True)[:, None, :, :].float()

    return log_mel_examples_tensor
