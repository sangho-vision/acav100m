import random
import torch
import torchvision
from pathlib import Path

import utils.logging as logging

import data.utils as utils
from data.build import DATASET_REGISTRY


logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class UCF101(torch.utils.data.Dataset):
    """
    UCF101 video loader. Construct the UCF101 video loader, then sample
    clips from the videos. For training and validation, multiple clips are
    uniformly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """
    def __init__(self, cfg, mode):
        """
        Construct the UCF101 video loader with given text files containing
        video paths and a list of classes.
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
        """
        assert mode in [
            "train",
            "val",
            "test",
        ], "Mode '{}' not supported for UCF101".format(mode)

        if mode == "val":
            logger.info(
                "UCF101 does not have the val split... "
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
            assert cfg.TEST.NUM_SAMPLES == cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS, \
                f"test num samples {cfg.TEST.NUM_SAMPLES} must be #views {cfg.TEST.NUM_ENSEMBLE_VIEWS} x #crops {cfg.TEST.NUM_SPATIAL_CROPS}"

        self.split = cfg[mode.upper()]["DATASET_SPLIT"]

        logger.info(f"Constructin UCF101 mode {self.mode} split {self.split}")
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        mode = 'test' if self.mode == 'val' else self.mode
        dir_to_files = Path(self.cfg.DATASET_DIR)

        class_file = dir_to_files.joinpath(
            "splits",
            "classInd.txt"
        )

        path_to_file = dir_to_files.joinpath(
            "splits",
            f"{mode}list{self.split:02d}.txt"
        )

        assert class_file.exists(), "{} not found".format(str(class_file))
        assert path_to_file.exists(), "{} not found".format(str(path_to_file))

        with open(class_file, "r") as f:
            classInd = f.readlines()

        self.class2idx = {
            l.strip().split()[1]: int(l.strip().split()[0]) - 1 for l in classInd
        }
        self.idx2class = [l.strip().split()[1] for l in classInd]

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []

        with open(path_to_file, "r") as f:
            dataset = f.readlines()
        dataset = [l.strip() for l in dataset]

        for clip_idx, l in enumerate(dataset):
            if mode in ['train']:
                path = str(dir_to_files.joinpath(l.split()[0]))
                label = self.class2idx[l.split()[0].split('/')[0]]
            elif mode in ['test']:
                path = str(dir_to_files.joinpath(l))
                label = self.class2idx[l.split('/')[0]]
            for idx in range(self._num_clips):
                self._path_to_videos.append(path)
                self._labels.append(label)
                self._spatial_temporal_idx.append(idx)

        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load UCF101 mode {} split {}".format(
            self.mode, self.split,
        )
        logger.info(
            "Constructing UCF101 dataloader (mode: {}, split: {}, size: {})".format(
                self.mode, self.split, len(self._path_to_videos),
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
        Given the video index, return the list of frames, label, and
        video index.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            visual_clip (tensor): the frames of sampled from the video.
                The dimension is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
        """
        frames, waveform, info = torchvision.io.read_video(
            self._path_to_videos[index],
            pts_unit="sec",
        )

        video_fps = round(info["video_fps"])
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

        delta = max(frames.size(0) - _num_frames, 0)
        # If num_samples == 1, a single clip is randomly sampled from the input
        # video. Otherwise, multiple clips are uniformly sampled.
        if num_samples > 1:
            start_idx = (
                delta * temporal_sample_index / (num_samples - 1)
            )
        else:
            start_idx = random.uniform(0, delta)
        end_idx = start_idx + _num_frames - 1
        visual_clip = self.get_visual_clip(
            frames,
            start_idx,
            end_idx,
            min_scale,
            max_scale,
            crop_size,
            spatial_sample_index,
        )

        label = self._labels[index]

        return visual_clip, label, index

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
        Args:
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
        Returns:
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
