import random
import torch
import torchvision
from pathlib import Path

import utils.logging as logging
from data.build import DATASET_REGISTRY
import data.transform as transform
import data.utils as utils


logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class KineticsSounds(torch.utils.data.Dataset):
    """
    Kinetics-Sounds video loader. Construct the Kinetics-Sounds video loader,
    then sample audio/visual clips from the videos. For training and validation,
    multiple audio/visual clips are uniformly sampled from every video with
    audio/visual random transformations. For testing, multiple audio/visual
    clips are uniformly sampled from every video with only uniform cropping.
    For uniform cropping, we take the left, center, and right crop
    if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """
    def __init__(self, cfg, mode):
        assert mode in [
            "train",
            "val",
            "test",
        ], "Mode {} not suported for KineticsSounds".format(mode)

        self.mode = mode
        self.cfg = cfg

        if self.mode in ['train', 'val']:
            self._num_clips = cfg.TRAIN.NUM_SAMPLES
        elif self.mode in ['test']:
            self._num_clips = (
                cfg.TEST.NUM_SAMPLES
            )
            assert cfg.TEST.NUM_SAMPLES == cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS, \
                f"test num samples {cfg.TEST.NUM_SAMPLES} must be #views {cfg.TEST.NUM_ENSEMBLE_VIEWS} x #crops {cfg.TEST.NUM_SPATIAL_CROPS}"

        logger.info(f"Constructin KineticsSounds mode {self.mode}")
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        mode = self.mode
        data_dir = Path(self.cfg.DATASET_DIR).joinpath(f"{mode}")

        self.idx2label = sorted(data_dir.iterdir())
        self.idx2label = [path.name for path in self.idx2label]
        self.label2idx = {label: idx for idx, label in enumerate(self.idx2label)}

        videos = sorted(data_dir.rglob("*.mp4"))

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []

        for video_idx, video_path in enumerate(videos):
            yid = video_path.stem
            path = str(video_path)
            label = self.label2idx[video_path.parent.name]
            for idx in range(self._num_clips):
                self._path_to_videos.append(path)
                self._labels.append(label)
                self._spatial_temporal_idx.append(idx)

        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load KineticsSounds mode {}".format(
            self.mode,
        )
        logger.info(
            "Constructing KineticsSounds dataloader (mode: {}, size: {})".format(
                self.mode, len(self._path_to_videos),
            )
        )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

    def __getitem__(self, index):
        """
        Given the video index, return the audio/visual clips, label and
        video index.
        args:
            index (int): the video index provided by the pytorch sampler.
        returns:
            visual_clip (tensor): the frames sampled from the video.
                The dimension is `channel` x `num frames` x `height` x `width`.
            audio_clip (tensor): log-mel-spectrogram sampled from the video.
                The dimension is `channel` x `frequency` x `time`.
            label (int): the label of the current video.
            index (int): the index of the video.
        """
        frames, waveform, info = torchvision.io.read_video(
            self._path_to_videos[index],
            pts_unit="sec",
        )

        video_fps = round(info["video_fps"])
        audio_fps = info["audio_fps"]
        # Convert waveform to a nonophonic signal, and resample to the
        # target sampling rate if needed.
        waveform = transform.resample(
            waveform, audio_fps, self.cfg.DATA.TARGET_AUDIO_RATE, use_mono=True,
        )
        if self.mode in ['train', 'val']:
            temporal_sample_index = self._spatial_temporal_idx[index]
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            num_samples = self._num_clips
        elif self.mode in ['test']:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )

            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            assert len({min_scale, max_scale, crop_size}) == 1
            num_samples = self.cfg.TEST.NUM_ENSEMBLE_VIEWS
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        # Adjust number of frames consdiering input video fps, taget fps and
        # frame sampling rate.
        _num_frames = (
            self.cfg.DATA.NUM_FRAMES *
            self.cfg.DATA.SAMPLING_RATE *
            video_fps /
            self.cfg.DATA.TARGET_FPS
        )
        # Compute audio waveform size corresponding to the visual clip.
        waveform_size = int(
            self.cfg.DATA.TARGET_AUDIO_RATE *
            self.cfg.DATA.NUM_FRAMES *
            self.cfg.DATA.SAMPLING_RATE /
            self.cfg.DATA.TARGET_FPS
        )

        visual_delta = max(frames.size(0) - _num_frames, 0)
        if num_samples > 1:
            visual_start_idx = (
                visual_delta * temporal_sample_index / (num_samples - 1)
            )
        else:
            visual_start_idx = random.uniform(0, visual_delta)
        visual_end_idx = visual_start_idx + _num_frames - 1
        visual_clip = self.get_visual_clip(
            frames,
            visual_start_idx,
            visual_end_idx,
            min_scale,
            max_scale,
            crop_size,
            spatial_sample_index,
        )

        audio_delta = max(waveform.size(-1) - waveform_size, 0)
        audio_start_idx = int(
            audio_delta * (visual_start_idx / visual_delta)
        )
        audio_end_idx = audio_start_idx + waveform_size
        audio_clip = self.get_audio_clip(
            waveform,
            audio_start_idx,
            audio_end_idx,
            True if self.mode in ["train"] else False,
        )

        label = self._labels[index]
        return visual_clip, audio_clip, label, index

    def get_visual_clip(
        self,
        frames,
        start_idx,
        end_idx,
        min_scale,
        max_scale,
        crop_size,
        spatial_sample_index,
    ):
        """
        Sample a clip from the input video, and apply visual transformations.
        args:
            frames (tensor): a tensor of video frames, dimension is
                `num frames` x `height` x `width` x `channel`.
            start_idx (float): the index of the start frame.
            end_idx (float): the index of the end frame.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
            spatial_sample_index (int): if -1, perform random spatial sampling.
                If 0, 1, or 2, perform left, center, right crop if width is
                larger than height, and perform top, center, buttom crop if
                height is larger than width.
        returns:
            clip (tensor): sampled frames. The dimension is
                `channel` x `num frames` x `height` x `width`.
        """
        # Temporal sampling.
        clip = utils.temporal_sampling(
            frames,
            start_idx,
            end_idx,
            self.cfg.DATA.NUM_FRAMES,
        )

        # Convert frames of the uint type in the range [0, 255] to
        # a torch.FloatTensor in the range [0.0, 1.0]
        clip = clip.float()
        clip = clip / 255.0

        # T H W C -> T C H W
        clip = clip.permute(0, 3, 1, 2)

        # Visual transformations.
        clip = utils.apply_visual_transform(
            self.cfg,
            clip,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
        )

        # T C H W -> C T H W
        clip = clip.transpose(0, 1).contiguous()
        clip = [clip]

        return clip

    def get_audio_clip(
        self,
        waveform,
        audio_start_idx,
        audio_end_idx,
        apply_transform=False,
    ):
        """
        Sample an audio clip from the input video and apply audio
        transformations.
        Args:
            waveform (tensor): a tensor of audio waveform, dimension is
                `channel` x `time`.
            audio_start_idx (int): the start index.
            audio_end_idx (int): the end index.
            apply_transform (bool): whether to apply transformations.
        Returns:
            (tensor): log-mel-scaled spectrogram with dimension of
                `channel` x `frequency` x `time`.
        """
        # Temporal sampling.
        waveform_view = waveform[:, audio_start_idx:audio_end_idx]
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
