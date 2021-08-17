import os, sys
import math
import random
import argparse
import subprocess
from itertools import product

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Clip extractor based on FFMPEG')
    parser.add_argument('-i', '--in_filepath', type=str, required=True, help='Input video filepath')
    parser.add_argument('-o', '--out_dir', type=str, default='.', help='Directory to save extracted clips')
    parser.add_argument('-n', '--num_clips', type=int, default=3, help='Number of desired output clips (def. 3)')
    parser.add_argument('-t', '--threshold', type=float, default=10.0, help='SBD threshold (%% of max change, [0., 100.]) (def. 10)')
    parser.add_argument('-d', '--clip_duration', type=float, default=10.0, help='Output clip duration in sec (def. 10))')
    parser.add_argument('--anneal_factor', type=float, default=1.2, help='Threshold annealing factor (def 1.1)')
    parser.add_argument('--force_duration', action='store_true', help='Force output clips to have CLIP_DURATION')
    parser.add_argument('--force_num_clips', action='store_true', help='Force extracting NUM_CLIPS')
    parser.add_argument('--sampling', type=str, default='random', choices=('random', 'diversity'), help='Shot sampling mechanism (def. random)')
    args = parser.parse_args()
    return args


def hhmmss(sec):
    hh = int(sec // 3600)
    ss = sec % 3600
    mm = int(ss // 60)
    ss = ss % 60
    return '{:02d}:{:02d}:{:f}'.format(hh, mm, ss)


def get_clip_duration(filepath):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', filepath]
    out = run(cmd)
    try:
        duration = float(out.strip())
    except:
        duration = -1.0
    return duration


def get_filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def run(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = proc.communicate()
    return out.decode('utf-8')


def run_sbd(filepath, threshold):
    cmd = ['ffmpeg', '-i', filepath, '-vf', 'scdet=threshold={}'.format(threshold),
           '-f', 'null', ' - 2>&1']
    out = run(cmd)
    sbd_results = [x.strip() for x in out.splitlines() if x.startswith('[scdet')]

    shot_boundaries = [float(x.split(':')[-1]) for x in sbd_results]
    boundary_scores = [float(x.split(':')[-2].split(',')[0]) for x in sbd_results]
    return shot_boundaries, boundary_scores


def get_valid_clips(sb, min_duration, force_duration=False):
    sb = [0] + sb
    shots = [[sb[i - 1], sb[i]] for i in range(1, len(sb))]
    valid_clips = [shots[i] for i in range(len(shots))
                   if shots[i][1] - shots[i][0] >= min_duration]

    if force_duration:
        for i in range(len(valid_clips)):
            sb = valid_clips[i]
            delta = .5 * ((sb[1] - sb[0]) - min_duration)
            valid_clips[i][0] = sb[0] + delta
            valid_clips[i][1] = valid_clips[i][0] + min_duration

    return valid_clips


def get_mean_clip(full_duration, min_duration):
    assert full_duration >= min_duration, "clip duration shorter than min duration"
    mean = full_duration / 2
    padding = min_duration / 2
    return [mean - padding, mean + padding]


def extract_clip(sb, in_filepath, out_filepath):
    cmd = ['ffmpeg', '-ss', hhmmss(sb[0]), '-i', in_filepath, '-t', hhmmss(sb[1] - sb[0]),
           '-c', 'copy', '-avoid_negative_ts', '1', '-reset_timestamps', '1',
           '-y', '-hide_banner', '-loglevel', 'panic', '-map', '0', out_filepath]
    run(cmd)
    if not os.path.isfile(out_filepath):
        raise Exception(f"{out_filepath}: ffmpeg clip extraction failed")


def compute_perceptual_similarity(video_0, video_1):
    cmd = ['ffmpeg', '-i', video_0, '-i', video_1, '-hide_banner',
           '-filter_complex', 'signature=detectmode=full:nb_inputs=2',
           '-f', 'null', ' - 2>&1']
    out = run(cmd)
    out = [x for x in out.split('\n') if 'Parsed_signature_0' in x and 'frames matching' in x]
    if not out:
        return 0.

    num_matched_frames = int(out[0].split(',')[-1].split()[0])
    return num_matched_frames


def calc_diversity(sim, num_clips, calc_sum=True):
    if calc_sum:
        return calc_sum_of_pairwise_distance(sim, num_clips)
    else:
        return calc_pairwise_distance(sim, num_clips)


def calc_pairwise_distance(sim, num_clips):
    # Greedy Algorithm for Minimum Pairwise Distance Metric
    keep_idx = [0]  # randomly choose the first point
    if num_clips == 1:
        return keep_idx
    for _ in range(num_clips - 1):
        row = np.argsort(sim[keep_idx[-1]])  # sort by minimum similarity
        row = np.setdiff1d(row, np.array(keep_idx))  # remove already chosen indices (including diagonals)
        current = row[0]
        keep_idx.append(current)
    return keep_idx


def calc_sum_of_pairwise_distance(sim, num_clips, eps=0.1, big_number=1e+10):
    # Local Search Algorithm for Sum of Pairwise Distance Metric
    gain_coeff = (1 + eps / sim.shape[0])

    '''
    def sum_diversity(indices):
        nonlocal sim
        options = list(product(indices, indices))
        return sum(sim[o[0]][o[1]] for o in options) / 2  # account for duplication
    '''

    # get argmin
    min_set = set(np.unravel_index(sim.argmin(), sim.shape))
    diff = num_clips - len(min_set)
    if diff <= 0:
        return list(min_set)[:num_clips]
    # populate the rest with random assignment
    rest = list(set(range(sim.shape[0])) - min_set)[:diff]
    current_set = list(set(rest) | set(min_set))
    assert len(current_set) == num_clips, "diversity calculation failed on initialization"

    swapped = True
    print("initiating swap")
    while swapped:
        swapped = False
        for i in range(num_clips):
            # calculate swap gain
            idx = current_set[i]
            rest = list(set(current_set) - set([idx]))
            rest_sum = sim[rest].sum(axis=0)
            rest_sum[rest] = big_number  # masking out already selected points
            min_idx = rest_sum.argmin()
            min_val = rest_sum[min_idx]
            current_val = rest_sum[idx]

            if gain_coeff * min_val < current_val:
                # swap
                print("swapping with gain {}".format(gain_coeff * current_val - min_val))
                current_set.remove(idx)
                current_set.append(min_idx)
                swapped = True
                break

    return list(current_set)


def main(in_filepath, out_dir, threshold=10.0,
         clip_duration_threshold=[60.0],
         clip_duration=10.0, force_duration=True,
         num_clips=3, force_num_clips=True,
         anneal_factor=1.2, sampling='random',
         cut_random_clips=None, calc_diversity_with_sum=False,
         **kwargs):
    # check if file exists
    if not os.path.isfile(in_filepath):
        sys.exit('No such file: {}'.format(in_filepath))

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    orig_clip_duration = get_clip_duration(in_filepath)
    # loosening force_num_clip constraint
    for i, constraint in enumerate(sorted(clip_duration_threshold)):
        if orig_clip_duration <= constraint:
            num_clips = math.ceil(num_clips / 2 ** (len(clip_duration_threshold) - 1))
            break
    if num_clips < 1:
        num_clips = 1

    threshold = float(threshold)
    done = False
    valid_clips = []
    while not done:
        # run shot boundary detector
        sb, sb_score = run_sbd(in_filepath, threshold)

        if len(sb) != 0:
            # get clips longer than clip_duration
            valid_clips = get_valid_clips(sb, clip_duration, force_duration)

        done = len(valid_clips) >= num_clips or not force_num_clips \
            or threshold >= 100.
        if not done:
            threshold = min(float(anneal_factor) * threshold, 100.)

    if len(valid_clips) == 0:
        # fallback to the mean clip
        du = get_clip_duration(in_filepath)
        sb = [0, du]
        if force_duration:
            delta = .5 * ((sb[1] - sb[0]) - clip_duration)
            sb = [sb[0] + delta, sb[0] + delta + clip_duration]
        valid_clips = [sb]
        num_clips = len(valid_clips)

    def save_clip(clip, idx=None):
        if idx is not None:
            out_filepath = os.path.join(
                out_dir,
                "{}_{:02d}.mp4".format(
                    get_filename(in_filepath), idx,
                )
            )
        else:
            out_filepath = os.path.join(
                out_dir,
                '{}_{:03d}.mp4'.format(
                    get_filename(in_filepath), int(clip[0]),
                )
            )
        if not os.path.isfile(out_filepath):
            extract_clip(clip, in_filepath, out_filepath)
        return out_filepath

    def compute_save_delete(path_as, clip_b):
        path_b = save_clip(clip_b)
        sim = 0
        for path_a in path_as:
            sim += compute_perceptual_similarity(path_a, path_b)
        os.remove(path_b)
        return sim, path_b

    # clip sampling: random (fast)
    if force_num_clips and len(valid_clips) > num_clips \
            and sampling == 'random':
        valid_clips = sorted(random.sample(valid_clips, num_clips))

    if sampling == 'diversity' and cut_random_clips is not None:
        assert cut_random_clips >= num_clips, "cut_random clips should be larger than num_clips"
        valid_clips = sorted(random.sample(valid_clips, num_clips))[:cut_random_clips]

    if sampling == 'diversity_greedy':
        # shuffle valid_clips
        random.shuffle(valid_clips)
        out_filepaths = []
        if len(valid_clips) <= num_clips:
            for idx, clip in enumerate(valid_clips):
                out_filepath = save_clip(clip)
                out_filepaths.append(out_filepath)
            num_clips = len(valid_clips)
            saved_clips = valid_clips
        else:
            current_clips = [valid_clips[0]]
            other_clips = valid_clips[1:]
            out_filepaths = [save_clip(current_clips[-1])]
            for i in range(num_clips - 1):
                # prepare files
                min_sim = 1e+10
                for i, other_clip in enumerate(other_clips):
                    sim, other_path = compute_save_delete(out_filepaths, other_clip)
                    if sim == 0:
                        clip_candidate = i
                        break
                    if sim < min_sim:
                        clip_candidate = i
                        min_sim = sim
                current_clip = other_clips[clip_candidate]
                current_clips = [*current_clips, current_clip]
                out_filepaths.append(save_clip(current_clips[-1]))
                other_clips.pop(clip_candidate)
            saved_clips = current_clips
        return saved_clips, out_filepaths

    # extract valid clips
    out_filepaths = []
    for clip in valid_clips:
        out_filepath = save_clip(clip)
        out_filepaths.append(out_filepath)

    # clip sampling: diversity (slow)
    keep_idx = list(range(len(valid_clips)))
    if force_num_clips and len(valid_clips) > num_clips:
        if sampling == 'diversity':
            n = len(valid_clips)
            sim = np.zeros((n, n))
            random.shuffle(out_filepaths)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    sim[i, j] = compute_perceptual_similarity(
                        out_filepaths[i], out_filepaths[j]
                    )
            sim = sim + sim.T  # get full matrix
            keep_idx = calc_diversity(sim, num_clips, calc_diversity_with_sum)
            [os.remove(out_filepaths[i]) for i in range(n) if i not in keep_idx]
        elif sampling in ['random_then_diversity', 'random1_then_diversity']:
            # random + diversity: middle ground?
            random.shuffle(out_filepaths)
            random_clips = math.ceil(num_clips / 2)
            if sampling == 'random1_then_diversity':
                random_clips = 1
            diversity_clips = num_clips - random_clips
            keep_idx = list(range(random_clips))  # out_filepaths are already shuffled
            n = len(valid_clips)
            sim = np.zeros((random_clips, n - random_clips))
            for i in range(random_clips):
                for j in range(n - random_clips):
                    sim[i, j] = compute_perceptual_similarity(
                        out_filepaths[i], out_filepaths[j + random_clips]
                    )
            diversity_keep_idx = (np.argsort(np.sum(sim, axis=0))[:diversity_clips] + random_clips)
            keep_idx = [*keep_idx, *list(diversity_keep_idx)]
            [os.remove(out_filepaths[i]) for i in range(n) if i not in keep_idx]

    saved_clips = [valid_clips[idx] for idx in keep_idx]
    out_filepaths = [out_filepaths[idx] for idx in keep_idx]
    return saved_clips, out_filepaths


if __name__ == '__main__':
    random.seed(98052)
    args = parse_args()
    main(**args.__dict__)
