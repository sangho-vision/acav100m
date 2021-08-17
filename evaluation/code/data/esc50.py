import os
import torch
import torchaudio
from pathlib import Path

import utils.logging as logging

import data.transform as transform
import data.utils as utils
from data.build import DATASET_REGISTRY


logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class ESC50(torch.utils.data.Dataset):
    """
    ESC-50 audio loader. Construct the ESC-50 audio loader, then sample
    clips from the audios. For training and validation, multiple clips are
    uniformly sampled from every audio with random masking. For testing,
    multiple clips are uniformaly sampled from every audio without any masking.
    We convert the input audio to a monophonic signal and convert it to
    log-mel-scaled spectrogram.
    """
    def __init__(self, cfg, mode):
        """
        Construct the ESC-50 audio loader.
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
        """
        assert mode in [
            "train",
            "val",
            "test",
        ], "Mode '{}' not supported for ESC50".format(mode)

        if mode == "val":
            logger.info(
                "ESC50 does not have the val split... "
                "Instead we will use the test split"
            )

        self.mode = mode
        self.cfg = cfg

        if self.mode in ['train', 'val']:
            self._num_clips = cfg.TRAIN.NUM_SAMPLES
        elif self.mode in ['test']:
            self._num_clips = (
                cfg.TEST.NUM_SAMPLES
            )

        self.split = cfg[mode.upper()]["DATASET_SPLIT"]

        logger.info(f"Constructin ESC50 mode {self.mode} split {self.split}")
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the audio loader.
        """
        dir_to_files = Path(
            os.path.join(
                self.cfg.DATASET_DIR,
                "audio",
            )
        )
        split_files = sorted(
            dir_to_files.glob("*.wav")
        )
        if self.mode in ["train"]:
            split_files = [
                path for path in split_files
                if int(path.name.split('.')[0].split('-')[0]) != self.split
            ]
        elif self.mode in ["val", "test"]:
            split_files = [
                path for path in split_files
                if int(path.name.split('.')[0].split('-')[0]) == self.split
            ]

        self._path_to_audios = []
        self._labels = []
        self._temporal_idx = []

        for clip_idx, path in enumerate(split_files):
            label = int(path.name.split('.')[0].split('-')[-1])
            for idx in range(self._num_clips):
                self._path_to_audios.append(str(path))
                self._labels.append(label)
                self._temporal_idx.append(idx)

        assert (
            len(self._path_to_audios) > 0
        ), "Failed to load ESC50 mode {} split {}".format(
            self.mode, self.split,
        )
        logger.info(
            "Constructing ESC50 dataloader (mode: {}, split: {}, size: {})".format(
                self.mode, self.split, len(self._path_to_audios),
            )
        )

    def __len__(self):
        """
        Returns:
            (int): the number of audios in the dataset.
        """
        return len(self._path_to_audios)

    def __getitem__(self, index):
        """
        Given the audio index, return the log-mel-scaled spectrogram, label,
        and audio index.
        args:
            index (int): the audio index provided by the pytorch sampler.
        returns:
            audio_clip (tensor): log-mel-spectrogram sampled from the audio.
                The dimension is `channel` x `frequency` x `time`.
            label (int): the label of the current audio.
            index (int): the index of the audio.
        """
        waveform, audio_fps = torchaudio.load(self._path_to_audios[index])
        # Convert it to a monophonic signal, and resample to the
        # target sampling rate if needed.
        waveform = transform.resample(
            waveform,
            audio_fps,
            self.cfg.DATA.TARGET_AUDIO_RATE,
            use_mono=True,
        )
        total_length = waveform.size(1)
        # We sample a `DATA.CLIP_DURATION`-sec clip.
        clip_length = (
            self.cfg.DATA.TARGET_AUDIO_RATE * self.cfg.DATA.CLIP_DURATION
        )
        delta = max(total_length - clip_length, 0)
        start_idx = int(
            delta * self._temporal_idx[index] / (self._num_clips - 1)
        )
        audio_clip = self.get_audio(
            waveform,
            start_idx,
            clip_length,
            True if self.mode in ['train'] else False,
        )
        label = self._labels[index]
        return audio_clip, label, index

    def get_audio(
        self,
        waveform,
        start_idx,
        clip_length,
        apply_transform=False,
    ):
        """
        Sample a clip from the input audio, and apply audio transformations.
        Args:
            waveform (tensor): a tensor of audio waveform, dimension is
                `channel` x `time`.
            start_idx (int): the start index.
            clip_length (int): the size of audio clip.
            apply_transform (bool): whether to apply transformations.
        Returns:
            (tensor): log-mel-scaled spectrogram with dimension of
                `channel` x `frequency` x `time`.
        """
        # Temporal sampling.
        waveform_view = waveform[:, start_idx:start_idx + clip_length]
        # Convert it to log-mel-scaled spectrogram.
        log_mel_spectrogram = transform.get_log_mel_spectrogram(
            waveform_view,
            self.cfg.DATA.TARGET_AUDIO_RATE,
            self.cfg.DATA.AUDIO_FREQUENCY,
            self.cfg.DATA.AUDIO_TIME,
        )
        # Apply transformations.
        if apply_transform:
            log_mel_spectrogram = utils.apply_audio_transform(
                log_mel_spectrogram,
                self.cfg.DATA.FREQUENCY_MASK_RATE,
                self.cfg.DATA.TIME_MASK_RATE,
            )

        return log_mel_spectrogram
