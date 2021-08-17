from pathlib import Path
import argparse
import shutil
import random

from tqdm import tqdm

from extract_clips import main as _run_single_file

def parse_args():
    parser = argparse.ArgumentParser(description='Clip extractor based on FFMPEG')
    parser.add_argument('-p', '--data-path', type=str, required=True, help='Input Directory')
    parser.add_argument('--sampling', type=str, default='diversity_greedy',
                        choices=('random', 'diversity',
                                 'diversity_greedy',
                                 'random_then_diversity',
                                 'random1_then_diversity'),
                        help='Shot sampling mechanism (def. random)')
    parser.add_argument('-c', '--cut-random-clips', type=int, required=None)
    parser.add_argument('--calc-diversity-with-sum', action='store_true')
    args = parser.parse_args()
    return args


def run_single_file(in_path, args):
    name = args.sampling
    if name == 'diversity':
        diversity = 'sum_of_pairwise' if args.calc_diversity_with_sum else 'minimum_pairwise'
        name = "{}_{}".format(name, diversity)
    out_dir = in_path.parent / "clips_{}".format(name)
    out_dir.mkdir(exist_ok=True, parents=True)
    out_dir = out_dir / in_path.stem
    if out_dir.is_dir():
        shutil.rmtree(str(out_dir))
    out_dir.mkdir(exist_ok=False, parents=True)
    return _run_single_file(str(in_path), str(out_dir), sampling=args.sampling,
                            cut_random_clips=args.cut_random_clips,
                            calc_diversity_with_sum=args.calc_diversity_with_sum)


def main(args):
    random.seed(98052)
    all_videos = sorted(list(Path(args.data_path).resolve().glob('*.mp4')))
    all_videos = [v for v in all_videos if v.name.count('.') == 1]
    all_videos = [v for v in all_videos if not v.parent.name.startswith('clips_')]
    total_num_clips = 0
    print("using sampling: {}".format(args.sampling))
    clip_timestamps = {}
    for video in tqdm(all_videos, total=len(all_videos)):
        saved_clips, _ = run_single_file(video, args)
        clip_timestamps[str(video)] = saved_clips
        total_num_clips += len(saved_clips)
    print("clips/videos: ({}/{})".format(total_num_clips, len(all_videos)))
    return clip_timestamps


if __name__ == '__main__':
    args = parse_args()
    main(args)
