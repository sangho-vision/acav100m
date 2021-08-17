import csv
from pathlib import Path
from collections import defaultdict


def save_output(data, metas, out_path, name='', sharded_meta=True):
    out_path.parent.mkdir(exist_ok=True, parents=True)
    data_with_metas = {}
    keys = []
    no_meta = 0
    for row in data:
        fname = Path(row['filename']).stem
        flag = False
        if sharded_meta:
            shard_name = row['shard_name']
            if shard_name in metas:
                if fname in metas[shard_name]:
                    meta = metas[shard_name][fname]
                    flag = True
        else:
            print(fname)
            if fname in metas:
                meta = metas[fname]
                flag = True
        if not flag:
            meta = {'id': '-1', 'segment': [-1.0, -1.0]}
            no_meta += 1
        row = {**row, **meta}
        data_with_metas[fname] = row
        keys.append(fname)

    # save as csv
    headers = ['shard_name', 'filename', 'id', 'segment']
    count = 0
    out_path = out_path.parent / (name + out_path.name)
    with open(out_path, 'a+') as f:
        writer = csv.writer(f)
        for key in keys:
            row = data_with_metas[key]
            line = [row[h] for h in headers]
            writer.writerow(line)
            count += 1

    return out_path, count


def format_rows(data, metas, sharded_meta=False,
                headers=['shard_name', 'filename', 'id', 'segment']):
    data_with_metas = {}
    keys = []
    no_meta = 0
    for row in data:
        fname = Path(row['filename']).stem
        flag = False
        if sharded_meta:
            shard_name = row['shard_name']
            if shard_name in metas:
                if fname in metas[shard_name]:
                    meta = metas[shard_name][fname]
                    flag = True
        else:
            if fname in metas:
                meta = metas[fname]
                flag = True
        if not flag:
            meta = {'id': '-1', 'segment': [-1.0, -1.0]}
            no_meta += 1
        row = {**row, **meta}
        data_with_metas[fname] = row
        keys.append(fname)

    # keep order
    # keys = sorted(keys)

    # save as csv
    # headers = ['shard_name', 'filename', 'id', 'segment']
    lines = []
    for key in keys:
        row = data_with_metas[key]
        line = [row[h] for h in headers]
        lines.append(line)
    return lines


def merge_csvs(ins, out):
    count = 0
    with open(out, 'a+') as out_f:
        for in_file in sorted(ins):
            with open(in_file, 'r') as in_f:
                for line in in_f:
                    out_f.write(line)
                    count += 1
    return count


def group_cache_paths(paths):
    groups = defaultdict(list)
    for path in paths:
        key = '_'.join(path.stem.split('_')[:2])
        groups[key].append(path)
    for k, v in groups.items():
        groups[k] = sorted(v)
    return groups


def merge_all_csvs(args):
    cache_dir = args.data.output.path.parent / 'caches'
    name = args.data.output.path.name
    paths = list(cache_dir.glob('cache_*_*_{}'.format(name)))
    groups = group_cache_paths(paths)

    for key in sorted(list(groups.keys())):
        print('processing cache set {}'.format(key))
        paths = groups[key]

        print("merging csvs")
        out_path = args.data.output.path
        counts = merge_csvs(paths, out_path)

        if args.verbose:
            print("Saved Results: added {} lines to {}".format(counts, out_path))

