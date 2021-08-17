import random
import tempfile

import torchvision

import data.transform as transform
import data.utils as utils


class VideoDecoder(object):
    """
    Video decoder for cross-modal (audio-visual) contrastive task. Sample
    audio/visual clips from the videos. Audio/visual clips are randomly
    sampled from every video with random transformations. Videos are stored in
    POSIX tar archives.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): configs.
        """
        self.cfg = cfg

    def decode(self, data):
        """
        Given video data, return the list of visual frames and
        the log-mel-scaled spectrogram.
        Args:
            data: video data.
        Returns:
            visual_clip (tensor): the visual frames sampled from the video.
                The dimension is `channel` x `num_frames` x `height` x `width`
            audio_clip (tensor): log-mel-spectrogram sampled from the video.
                The dimension is `channel` x `frequency` x `time`.
        """
        fname, video = data['__key__'], data['mp4']
        with tempfile.TemporaryDirectory() as dname:
            with open(dname+"/sample.mp4", "wb") as stream:
                stream.write(video)
            frames, waveform, info = \
                torchvision.io.read_video(
                    dname+"/sample.mp4",
                    pts_unit="sec",
                )
        video_fps = round(info["video_fps"])
        audio_fps = info["audio_fps"]
        # Convert waveform to a nonophonic signal, and resample to the
        # target sampling rate if needed.
        waveform = transform.resample(
            waveform, audio_fps, self.cfg.DATA.TARGET_AUDIO_RATE, use_mono=True,
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

        # Random audio/visual clip sampling.
        visual_delta = max(frames.size(0) - _num_frames, 0)
        audio_delta = max(waveform.size(-1) - waveform_size, 0)
        visual_start_idx = random.uniform(0, visual_delta)
        visual_end_idx = visual_start_idx + _num_frames - 1
        visual_clip = self.get_visual_clip(
            frames,
            visual_start_idx,
            visual_end_idx,
        )
        audio_start_idx = int(
            audio_delta * (visual_start_idx / visual_delta)
        )
        audio_end_idx = audio_start_idx + waveform_size
        audio_clip = self.get_audio_clip(
            waveform,
            audio_start_idx,
            audio_end_idx,
        )

        return visual_clip, audio_clip

    def get_visual_clip(
        self,
        frames,
        visual_start_idx,
        visual_end_idx,
    ):
        """
        Sample a visual clip from the input video and apply visual
        transformations.
        Args:
            frames (tensor): a tensor of video frames, dimension is
                `num frames` x `height` x `width` x `channel`.
            visual_start_idx (float): the index of the start frame.
            visual_end_idx (float): the index of the end frame.
        Returns:
            clip (tensor): sampled frames. The dimension is
                `channel` x `num frames` x `height` x `width`.
        """
        min_scale = self.cfg.DATA.PRETRAIN_JITTER_SCALES[0]
        max_scale = self.cfg.DATA.PRETRAIN_JITTER_SCALES[1]
        crop_size = self.cfg.DATA.PRETRAIN_CROP_SIZE

        # Temporal sampling.
        clip = utils.temporal_sampling(
            frames,
            visual_start_idx,
            visual_end_idx,
            self.cfg.DATA.NUM_FRAMES,
        )

        # Convert frames of the uint type in the range [0, 255] to
        # a torch.FloatTensor in the range [0.0, 1.0]
        clip = clip.float()
        clip = clip / 255.0

        # T H W C -> T C H W
        clip = clip.permute(0, 3, 1, 2)

        # Visual Transformations.
        # -1 indicates random spatial sampling.
        clip = utils.apply_visual_transform(
            self.cfg,
            clip,
            spatial_idx=-1,
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
    ):
        """
        Sample an audio clip from the input video and apply audio
        transformations.
        Args:
            waveform (tensor): a tensor of audio waveform, dimension is
                `channel` x `time`.
            audio_start_idx (int): the start index.
            audio_end_idx (int): the end index.
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
        log_mel_spectrogram = utils.apply_audio_transform(
            log_mel_spectrogram,
            self.cfg.DATA.FREQUENCY_MASK_RATE,
            self.cfg.DATA.TIME_MASK_RATE,
        )

        return log_mel_spectrogram
