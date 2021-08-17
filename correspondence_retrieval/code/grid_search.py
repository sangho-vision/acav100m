import argparse
from multiprocessing import Pool
from pathlib import Path
from itertools import chain, product
import json

import numpy as np
import torch
from tqdm import tqdm

from cli import Cli

from pair_weights import _get_weights


def callback(*args):
    # results.append(*args)
    pass


def use_gpu(grid):
    return grid[1].get('clustering_func_type', 'none').startswith('sgd') or \
        grid[1].get('use_gpu', False)


def run_gpus(grids):
    num_gpus = torch.cuda.device_count()
    grid_chunks = list([grids[x:x+num_gpus] for x in range(0, len(grids), num_gpus)])
    for i, chunk in tqdm(enumerate(grid_chunks), total=len(grid_chunks), desc="chunk iters"):
        torch.multiprocessing.spawn(
            run_nodistributed,
            nprocs=len(chunk),
            args=(
                run,
                chunk,
            ),
            daemon=False,
            join=True
        )


def run_nodistributed(
    local_rank, func, grids
):
    torch.multiprocessing.freeze_support()
    torch.cuda.set_device(local_rank)
    grid = grids[local_rank]
    func(*grid)


def run(i, args):
    cli = Cli()
    print('running expr_{}'.format(i), flush=True)
    '''
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            '''
    args['verbose'] = False
    res = cli.run(**args)
    print('done expr_{}'.format(i), flush=True)
    return args, res


def main():
    torch.multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-g', '--grid-path', type=str, default='search_targets/default.json')
    parser.add_argument('-l', '--large', action='store_true')
    args = parser.parse_args()


    '''
    grids = [
        [
            {'nexprs': 1}
        ],
        [
            {'data_name': 'image_pair_rotation', 'shuffle_datapoints': False},
            {'data_name': 'image_pair_rotation', 'shuffle_datapoints': True},
            {'data_name': 'image_pair_flip', 'shuffle_datapoints': False},
            {'data_name': 'image_pair_flip', 'shuffle_datapoints': True},
            {'data_name': 'image_pair_mnist_sound', 'shuffle_datapoints': True},
            {'data_name': 'kinetics_sounds', 'shuffle_datapoints': False},
        ], [
            {'cluster_paring': 'diagonal'},
            {'cluster_paring': 'bipartite'},
            {'cluster_paring': 'combination'},
        ], [
            {'extract_each_layer': True}
        ], [
            {'ncentroids': 10},
            #{'ncentroids': 20},
            #{'ncentroids': 50},
        ], [
            {'start_indices_selection': 'zero'}
        ], [
            {'sample_level': False},
        ]
    ]
    '''

    path = Path(args.grid_path)
    if path.is_dir():
        all_grids = []
        paths = list(path.glob('*.json'))
        for path in paths:
            with open(path, 'r') as f:
                grids = json.load(f)
            all_grids.append(grids)
    else:
        with open(path, 'r') as f:
            grids = json.load(f)
        all_grids = [grids]

    def check_weight_type_overflow_ok(grid):
        if 'weight_type' in grid and grid['weight_type'] is not None:
            weights = _get_weights(5, grid['weight_type'])
            pairing = weights ** 2  # simulate pairing
            flags = np.isnan(pairing ** 2)  # assume high bound
            if np.any(flags):
                print(f"overflow in option: {grid['weight_type']}, {weights}")
                print("skipping the potential overflow exp")
                return False
        return True

    def process_grids(grids):
        if args.large:
            grids.append([{'nsamples_per_class': 1000}])
        else:
            grids.append([{'nsamples_per_class': 100}])

        grids = [(i, {k: v for d in options for k, v in d.items()})
                 for i, options in enumerate(product(*grids))]

        grids = [(i, dt) for i, dt in grids if check_weight_type_overflow_ok(dt)]
        return grids

    grids = list(chain(*[process_grids(grids) for grids in all_grids]))
    num_expr = len(grids)
    print("Running {} configs".format(num_expr))

    results = []

    devices = torch.cuda.device_count()
    with Pool(50) as p:
        gpu_grids = []
        mult_grids = []
        for grid in grids:
            if use_gpu(grid) and devices > 0:
                gpu_grids.append(grid)
            else:
                mult_grids.append(grid)
        if len(mult_grids) > 1:
            for grid in mult_grids:
                x = p.apply_async(run, args=grid, callback=callback)
        elif len(mult_grids) > 0:
            run(*mult_grids[0])
            # x.get()  # for debugging msg
        if len(gpu_grids) > 1:
            print('running in gpus')
            run_gpus(gpu_grids)
        elif len(gpu_grids) > 0:
            print('running in gpus')
            run(*gpu_grids[0])

        p.close()
        p.join()
        print(results)


    print("completed")


if __name__ == "__main__":
    main()
