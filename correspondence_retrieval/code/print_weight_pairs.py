import argparse
import json
from pathlib import Path
from itertools import chain, product

from pair_weights import _get_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-g', '--grid-path', type=str, default='search_targets/default.json')
    args = parser.parse_args()
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

    def process_grids(grids):
        grids = [(i, {k: v for d in options for k, v in d.items()})
                 for i, options in enumerate(product(*grids))]

        grids = [(i, dt) for i, dt in grids]
        return grids

    grids = list(chain(*[process_grids(grids) for grids in all_grids]))
    weight_types = list(set(v[1].get('weight_type', None) for v in grids))

    w = {k: list(_get_weights(5, weight_type=k)) for k in weight_types if k is not None}
    w = {k: w[k] for k in sorted(list(w.keys()))}

    with open('weight_types.json', 'w') as f:
        json.dump(w, f, indent=4)


if __name__ == '__main__':
    main()
