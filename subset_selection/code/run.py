from dataloader import load_data, preprocess
from run_greedy import run_greedy
from save import save_output


def _run(args, path):
    partitions, metas = load_data(path, args.data.meta.path,
                                  args.computation.num_workers, args.verbose,
                                  args.log_every)
    pbar = sorted(list(partitions.keys()))
    results = []
    for k in pbar:
        print('running partition {}/{}'.format(k, len(partitions.keys())))
        partition = partitions[k]
        samples_list = run_partition(args, partition, metas)
        results.append(samples_list)
    return results, metas


def run_single(args):
    results, metas = _run(args, args.data.path)
    counts = 0

    out_path = None
    for samples_list in results:
        out_path, count = save_output(samples_list, metas, args.data.output.path)
        counts += count

    if out_path is None:
        print("No files saved")

    if args.verbose:
        print("Saved Results: added {} lines to {}".format(counts, out_path))


def run_partition(args, data, metas):
    assignments, shard_names, filenames, clustering_types = preprocess(data,
                                                                       args.computation.num_workers,
                                                                       args.log_every,
                                                                       verbose=args.verbose)
    samples_list = run_greedy(args, assignments, shard_names, filenames, clustering_types,
                              args.subset.size,
                              args.subset.ratio,
                              measure_name=args.measure_name,
                              cluster_pairing=args.clustering.pairing,
                              shuffle_candidates=args.shuffle_candidates,
                              verbose=args.verbose)
    return samples_list
