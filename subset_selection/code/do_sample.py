from pathlib import Path
import subprocess
from collections import defaultdict
import multiprocessing
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--path', default='../data/caches/output_contrastive_temp_contrastive_inferred_cache_186629', type=str)
    parser.add_argument('-k', '--k', default=1000, type=int)
    parser.add_argument('-w', '--num-workers', default=40, type=int)
    args = parser.parse_args()
    return args


def check_output(cmd):
    if isinstance(cmd, list):
        cmd = ' '.join(cmd)
    ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()
    return output[0].decode('ascii')


def sort_csv(out_path, in_name, num_workers):
    in_path = out_path.parent / in_name
    out_path = out_path.parent / ('sorted_' + in_name)
    if out_path.is_file():
        print("file exists already {}".format(out_path))
        return out_path

    buffer_percent = round(60 / num_workers)
    output = check_output("sort --parallel={} --buffer-size={}% -t , -u -k 1,1gr -k 2 {} > {}".format(num_workers, buffer_percent, in_path, out_path))
    if len(output.strip()) < 1:
        print("sorted {}".format(in_path))
    else:
        print("error in sorting {}".format(in_path))
        print(output)
    return out_path


def remove_scores(out_path, in_name, k):
    in_path = out_path.parent / ('top_{}_'.format(k) + in_name)
    out_path = out_path.parent / ('top_{}_scoreless_'.format(k) + in_name)

    awk_opt = "'{print substr($0, index($0,$2))}'"
    output = check_output("awk -F, {} {} > {}".format(awk_opt, in_path, out_path))
    if len(output.strip()) < 1:
        print("removed scores from {}".format(in_path))
    else:
        print("error in removing scores from {}".format(in_path))
        print(output)
    return out_path


def remove_duplicates(out_path, in_name):
    in_path = out_path.parent / ('sorted_' + in_name)
    out_path = out_path.parent / ('unique_' + in_name)
    if out_path.is_file():
        print("file exists already {}".format(out_path))
        return out_path

    # output = check_output("awk -F, 'NR==FNR{a[$1]++; next} a[$1]>1 {} > {}".format(in_path, out_path))
    awk_opt = "'!visited[$0]++'"
    output = check_output("awk {} {} > {}".format(awk_opt, in_path, out_path))
    if len(output.strip()) < 1:
        print("removed duplicates from {}".format(in_path))
    else:
        print("error in removing duplicates from {}".format(in_path))
        print(output)
    return out_path


def remove_duplicates_single(in_path, out_path):
    awk_opt = "'!visited[$0]++'"
    output = check_output("awk {} {} > {}".format(awk_opt, in_path, out_path))
    if len(output.strip()) < 1:
        print("removed duplicates from {}".format(in_path))
    else:
        print("error in removing duplicates from {}".format(in_path))
        print(output)
    return out_path


def cut_top(out_path, in_name, k):
    in_path = out_path.parent / ('unique_' + in_name)
    out_path = out_path.parent / ('top_{}_'.format(k) + in_name)

    print('cutting {}'.format(out_path))
    output = check_output("head -n {} {} > {}".format(k, in_path, out_path))
    if len(output.strip()) < 1:
        print("cut top {} for {}".format(k, in_path))
    else:
        print("error in cutting top {} for {}".format(k, in_path))
        print(output)
    return out_path


def group_with_pid(caches):
    res = defaultdict(list)
    for path in caches:
        name = '_'.join(path.stem.split('_')[:-1])[:-1]
        res[name].append(path)
    return res


def get_cache_paths(args):
    output_name = args.path.stem
    name = "{}_*.csv".format(output_name)
    cache_dir = args.path.parent
    caches = cache_dir.glob(name)
    caches = group_with_pid(caches)
    cache_name, paths = list(caches.items())[0]  # should be single item
    '''
    max_len = max([len(v) for v in caches.values()])
    caches = {k: v for k, v in caches.items() if len(v) == max_len}
    cache_name, paths = list(caches.items())[0]
    '''

    print("loading from cache {}".format(cache_name))
    print("{} total cache files".format(len(paths)))
    return paths


def merge_csvs(paths, out_path, name):
    paths = [p.parent / (name + p.name) for p in paths]
    out_path = str(out_path)
    paths = ' '.join([str(p) for p in paths])
    output = check_output("cat {} > {}".format(paths, out_path))
    if len(output.strip()) < 1:
        print(output)
    return out_path


def run(args):
    args.path = Path(args.path)
    paths = get_cache_paths(args)
    with multiprocessing.Pool(len(paths)) as p:
        procs = []
        for path in paths:
            print("sorting {}".format(path))
            proc_args = (args.path, path.name, args.num_workers)
            procs.append(p.apply_async(sort_csv, args=proc_args))
        for proc in procs:
            proc.wait()
        procs = []
        for path in paths:
            print("removing duplicates for {}".format(path))
            proc_args = (args.path, path.name)
            procs.append(p.apply_async(remove_duplicates, args=proc_args))
        for proc in procs:
            proc.wait()
        procs = []
        for path in paths:
            print("cutting top {} for {}".format(args.k, path))
            proc_args = (args.path, path.name, args.k)
            procs.append(p.apply_async(cut_top, args=proc_args))
        for proc in procs:
            proc.wait()
        procs = []
        for path in paths:
            print("removing scores for {}".format(path))
            proc_args = (args.path, path.name, args.k)
            procs.append(p.apply_async(remove_scores, args=proc_args))
        for proc in procs:
            proc.wait()

    name = args.path.stem + '.csv'
    out_path = args.path.parent / 'duplicate_top_{}_{}'.format(args.k, name)
    out_path = merge_csvs(paths, out_path, 'top_{}_scoreless_'.format(args.k))
    in_path = out_path
    out_path = args.path.parent / 'top_{}_{}'.format(args.k, name)
    out_path = remove_duplicates_single(in_path, out_path)
    print("done {}".format(out_path))


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
