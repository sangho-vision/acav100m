from pathlib import Path
# from subprocess import check_output as _check_output
import subprocess
from collections import defaultdict
import shutil


'''
bash command requirements:
- awk
- sort
- cat
'''


def check_output(cmd):
    if isinstance(cmd, list):
        cmd = ' '.join(cmd)
    ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()
    return output[0].decode('ascii')


def load_row(row):
    index = row.find(',')
    row = row[index + 1:]
    splitted = row.split(',')
    score = splitted[0]
    filename = splitted[1]
    key = (float(score), filename)
    return key, row


def merge_csvs(paths, out_path, name):
    out_path = out_path.parent / 'caches' / (name + '_' + out_path.name)
    out_path = str(out_path)
    paths = ' '.join([str(p) for p in paths])
    output = check_output("cat {} > {}".format(paths, out_path))
    if len(output.strip()) < 1:
        print(output)
    return out_path


def sort_csv(out_path, in_name, out_name):
    in_path = out_path.parent / 'caches' / (in_name + '_' + out_path.name)
    out_path = out_path.parent / 'caches' / (out_name + '_' + out_path.name)

    output = check_output("sort -t , -u -k 1,1gr -k 2 {} > {}".format(in_path, out_path))
    if len(output.strip()) < 1:
        print("sorted {}".format(in_path))
    else:
        print("error in sorting {}".format(in_path))
        print(output)
    return out_path


def remove_scores(out_path, in_name, out_name):
    in_path = out_path.parent / 'caches' / (in_name + '_' + out_path.name)
    out_path = out_path.parent / 'caches' / (out_name + '_' + out_path.name)

    awk_opt = "'{print substr($0, index($0,$2))}'"
    output = check_output("awk -F, {} {} > {}".format(awk_opt, in_path, out_path))
    if len(output.strip()) < 1:
        print("removed scores from {}".format(in_path))
    else:
        print("error in removing scores from {}".format(in_path))
        print(output)
    return out_path


def remove_duplicates(out_path, in_name, out_name):
    in_path = out_path.parent / 'caches' / (in_name + '_' + out_path.name)
    out_path = out_path.parent / 'caches' / (out_name + '_' + out_path.name)

    # output = check_output("awk -F, 'NR==FNR{a[$1]++; next} a[$1]>1 {} > {}".format(in_path, out_path))
    awk_opt = "'!visited[$0]++'"
    output = check_output("awk {} {} > {}".format(awk_opt, in_path, out_path))
    if len(output.strip()) < 1:
        print("removed duplicates from {}".format(in_path))
    else:
        print("error in removing duplicates from {}".format(in_path))
        print(output)
    return out_path


def group_with_pid(caches):
    res = defaultdict(list)
    for path in caches:
        name = '_'.join(path.stem.split('_')[:-1])[:-1]
        res[name].append(path)
    return res


def get_cache_paths(args):
    output_name = Path(args.data.output.path).stem
    name = "{}_contrastive_inferred_cache_*_*.csv".format(output_name)
    cache_dir = args.data.output.path.parent / 'caches'
    caches = cache_dir.glob(name)
    caches = group_with_pid(caches)
    max_len = max([len(v) for v in caches.values()])
    caches = {k: v for k, v in caches.items() if len(v) == max_len}
    cache_name, paths = list(caches.items())[0]
    print("loading from cache {}".format(cache_name))
    print("{} total cache files".format(len(paths)))
    return paths


def merge_contrastive(args):
    out_path = Path(args.data.output.path)
    paths = get_cache_paths(args)
    print("merging csvs")
    path = merge_csvs(paths, out_path, 'merged')
    paths = []
    paths.append(path)
    print("sorting csv")
    path = sort_csv(out_path, 'merged', 'sorted')
    paths.append(path)
    print("removing scores")
    path = remove_scores(out_path, 'sorted', 'scoreless')
    paths.append(path)
    print("removing duplicates")
    path = remove_duplicates(out_path, 'scoreless', 'unique')
    paths.append(path)
    print("move output")
    shutil.copy(paths[-1], out_path)
    '''
    for path in paths:
        Path(path).unlink()
    '''
    print("done")
