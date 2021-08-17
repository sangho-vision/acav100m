import argparse
import json

from cli import Cli


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-g', '--path', type=str, default='search_targets/default.json')
    parser.add_argument('-l', '--large', action='store_true')
    args = parser.parse_args()

    with open(args.path, 'r') as f:
        options = json.load(f)
    if args.large:
        options['nsamples_per_class'] = 1000
    else:
        options['nsamples_per_class'] = 100

    run(options)


def run(args):
    cli = Cli()
    args['verbose'] = True
    res = cli.run(**args)
    return args, res


if __name__ == '__main__':
    main()
