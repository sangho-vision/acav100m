import tarfile
from multiprocessing import Pool
import numpy as np
import subprocess
import os
from pathlib import Path
import math
import time
from datetime import datetime
from functools import partial
import argparse
from collections import defaultdict, Counter
import json
import pickle
import csv
import shutil

from tqdm import tqdm
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="construct ms100m dataset")
    parser.add_argument("--feat_dir", type=str, required=True, help="feature directory")
    # parser.add_argument("--new_feat_dir", type=str, required=True, help="new feature directory")
    parser.add_argument("--input_dir", type=str, required=True, help="input directory")
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    return args


def run(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = proc.communicate()
    return out.decode('utf-8')


def get_filenames_of_shard(shard_path):
    cmd = [
        "tar", "tvf", shard_path,
    ]
    out = run(cmd)
    out = out.strip().split('\n')
    filenames = [Path(f.split()[-1]).name for f in out if f.endswith('mp4')]
    return filenames


def indexing(pkl_path):
    shard_name = pkl_path.stem
    with open(pkl_path, "rb") as f:
        feats_shard = pickle.load(f)

    filenames = [feat['filename'] for feat in feats_shard]

    return shard_name, filenames


if __name__ == "__main__":
    args = parse_args()
    print(args)

    print("datset feature dir:  {}".format(args.feat_dir))

    feats_path = sorted(Path(args.feat_dir).glob("*.pkl"))

    print("dataset indexing start...")
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(indexing, feats_path),
                    ncols=80,
                    total=len(feats_path),
                )
            )
    else:
        results = []
        for pkl_path in tqdm(feats_path, total=len(feats_path), ncols=80):
            results.append(indexing(pkl_path))

    dataset_dict = {}
    for shard_name, filenames in results:
        dataset_dict[shard_name] = filenames

    print("dataset indexing done...")
    input_shard_names = sorted(list(dataset_dict.keys()))
    duplicate_files = defaultdict(list)
    print("duplicate checking...")
    for shard_name in tqdm(input_shard_names, ncols=80):
        if len(set(dataset_dict[shard_name])) != len(dataset_dict[shard_name]):
            filename_counter = Counter(dataset_dict[shard_name])
            for filename in set(dataset_dict[shard_name]):
                if filename_counter[filename] > 1:
                    duplicate_files[shard_name] += [(filename, filename_counter[filename])]

    num_duplicate_files = sum([len(duplicate_files[shard_name]) for shard_name in duplicate_files])
    print(f"# of duplicate files: {num_duplicate_files}")
    with open("dulicate_files.pkl", "wb") as f:
        pickle.dump(duplicate_files, f)


    non_matching_files = defaultdict(list)
    feat_dir = Path(args.feat_dir)
    new_feat_dir = Path(args.new_feat_dir)
    for shard_name in tqdm(input_shard_names, ncols=80):
        with open(os.path.join(args.input_dir, f"{shard_name}.json"), "r") as j:
            meta_shard = json.load(j)
        filenames = [meta['filename'] for meta in meta_shard]
        for filename in dataset_dict[shard_name]:
            if filename not in filenames:
                non_matching_files[shard_name] += [filename]

    num_non_matching_files = sum([len(non_matching_files[shard_name]) for shard_name in non_matching_files])

    print(f"# of non matching files: {num_non_matching_files}")
    with open("non_matching_files.pkl", "wb") as f:
        pickle.dump(non_matching_files, f)

    '''
    if len(non_matching_files) > 0:
        print(f"deleting non matching files")
        new_feat_dir.mkdir(exist_ok=True, parents=True)
        for shard_name in tqdm(input_shard_names, ncols=80):
            pkl_path = feat_dir.joinpath(f"{shard_name}.pkl")
            new_pkl_path = new_feat_dir.joinpath(f"{shard_name}.pkl")
            if shard_name in non_matching_files:
                with open(pkl_path, "rb") as f:
                    feats = pickle.load(f)
                new_feats = [feat for feat in feats if feat['filename'] not in non_matching_files[shard_name]]
                with open(new_pkl_path, "wb") as f:
                    pickle.dump(new_feats, f)
            else:
                shutil.copy(str(pkl_path), new_pkl_path)
    '''
