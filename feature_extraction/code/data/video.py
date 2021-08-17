import tempfile
import math
import numbers
from pathlib import Path

import torch
import torchvision
import numpy as np
import av


def load_video_webdata(data, num_frames, duration=None, skip_shorter_seconds=None):
    with tempfile.TemporaryDirectory() as dname:
        path = dname + "/sample.mp4"
        with open(path, "wb") as stream:
            stream.write(data)
        res = load_video(path, num_frames, duration, skip_shorter_seconds=skip_shorter_seconds)
    return res


def load_video(path, num_frames, duration=None, skip_shorter_seconds=None):
    # visual: (NumFrames, Height, Width, VideoChannels)
    # audio: (AudioChannels, Points)
    duration = {'start_pts': 0, 'end_pts': duration} if duration is not None else {}
    pts_unit = 'sec'
    visual, audio, info = read_video(str(path), pts_unit=pts_unit,
                                    **duration)
    if visual is None:
        return None
    if isinstance(skip_shorter_seconds, numbers.Number):
        num_seconds = visual.shape[0] / info['video_fps']
        if num_seconds < skip_shorter_seconds:
            return None
    visual, indices = temporal_sampling(visual, num_frames)
    audio = audio.transpose(0, 1)  # channel last
    res_visual = (None, None)
    res_audio = (None, None)
    if 'video_fps' in info:
        original_video_fps = info['video_fps']
        sampled_video_fps = original_video_fps / num_frames  # approximate
        res_visual = (visual, sampled_video_fps)
    if 'audio_fps' in info:
        res_audio = (audio, info['audio_fps'])
    return res_visual, res_audio


def save_video(data, path, fps):
    torchvision.io.write_video(path, data, fps)


def temporal_sampling(visual, num_frames):
    end_idx = visual.shape[0] - 1
    indices = torch.linspace(0, end_idx, num_frames).long()
    visual = torch.index_select(visual, 0, indices)
    return visual, indices


def test_video(path, num_frames):
    path = Path(path)
    (visual, fps), _ = load_video(path, num_frames)
    new_path = path.parent / f"test_{path.name}"
    save_video(visual, path, fps)
    print(f"{path} to {new_path}")


def read_video(
        filename: str, start_pts: int = 0, end_pts=None, pts_unit: str = "pts"
    ):
    '''
    from torchvision import get_video_backend

    if get_video_backend() != "pyav":
        return _video_opt._read_video(filename, start_pts, end_pts, pts_unit)

    _check_av_available()
    '''

    if end_pts is None:
        end_pts = float("inf")

    if end_pts < start_pts:
        raise ValueError(
            "end_pts should be larger than start_pts, got "
            "start_pts={} and end_pts={}".format(start_pts, end_pts)
        )

    info = {}
    video_frames = []
    audio_frames = []

    try:
        with av.open(filename, metadata_errors="ignore") as container:
            video_fps = None
            if container.streams.video:
                video_frames = torchvision.io.video._read_from_stream(
                    container,
                    start_pts,
                    end_pts,
                    pts_unit,
                    container.streams.video[0],
                    {"video": 0},
                )
                video_fps = container.streams.video[0].average_rate

            info["video_fps"] = float(video_fps)

            if container.streams.audio:
                time_base = container.streams.audio[0].time_base
                audio_frames = torchvision.io.video._read_from_stream(
                    container,
                    start_pts,
                    end_pts,
                    pts_unit,
                    container.streams.audio[0],
                    {"audio": 0},
                )

            info["audio_fps"] = container.streams.audio[0].rate
    #except av.AVError:
    #    pass
    except Exception as e:
        print(e.message, e.args)
        return None, None, {"video_fps": None, "audio_fps": None}

    vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]
    aframes_list = [frame.to_ndarray() for frame in audio_frames]

    if vframes_list:
        vframes = torch.as_tensor(np.stack(vframes_list))
    else:
        vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

    if aframes_list:
        aframes = np.concatenate(aframes_list, 1)
        aframes = torch.as_tensor(aframes)
        # fixing THIS line
        aframes = _align_audio_frames(aframes, audio_frames,
                                      *get_offsets(pts_unit, time_base, start_pts, end_pts))
    else:
        aframes = torch.empty((1, 0), dtype=torch.float32)

    return vframes, aframes, info


def get_offsets(pts_unit, time_base, start_offset, end_offset):
    if pts_unit == "sec":
        start_offset = int(math.floor(start_offset * (1 / time_base)))
        if end_offset != float("inf"):
            end_offset = int(math.ceil(end_offset * (1 / time_base)))
    return start_offset, end_offset


def _align_audio_frames(aframes, audio_frames, ref_start, ref_end):
    start, end = audio_frames[0].pts, audio_frames[-1].pts
    total_aframes = aframes.shape[1]
    step_per_aframe = (end - start + 1) / total_aframes
    s_idx = 0
    e_idx = total_aframes
    if start < ref_start:
        s_idx = int((ref_start - start) / step_per_aframe)
    if end > ref_end:
        e_idx = int((ref_end - end) / step_per_aframe)
    return aframes[:, s_idx:e_idx]
