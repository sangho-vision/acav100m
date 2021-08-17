import time
import datetime
from pathlib import Path

import fire

from args import get_args
from run import run_single
from run_contrastive import run_single_contrastive
from chunk import run_chunks, reduce_all_pkls
from chunk_contrastive import run_chunks_contrastive
from save import merge_all_csvs
from merge_contrastive import merge_contrastive
from tests import compare_measures


class Cli:
    def prepare(self, **kwargs):
        args = get_args(**kwargs)
        if 'out_path' in kwargs:
            args.data.output.path = Path(kwargs['out_path'])

        opath = args.data.output.path
        if opath.stem == opath.name:
            # potential dir
            opath = opath / 'output.csv'
        opath.parent.mkdir(parents=True, exist_ok=True)
        args.data.output.path = opath

        if 'shards_path' in kwargs:
            args.data.path = Path(kwargs['shards_path'])
        if 'meta_path' in kwargs:
            args.data.meta.path = Path(kwargs['meta_path'])

        mpath = args.data.meta.path
        if mpath is None:
            # use shard directory
            mpath = args.data.path.parent
        if not mpath.is_dir() and mpath.parent.is_dir():
            mpath = mpath.parent
        args.data.meta.path = mpath

        return args

    def run(self, **kwargs):
        start = time.time()
        args = self.prepare(**kwargs)
        run(args)
        elasped = time.time() - start
        elasped = str(datetime.timedelta(seconds=elasped))
        print('done. total time elasped: {}'.format(elasped))

    def reduce_csvs(self, **kwargs):
        start = time.time()
        args = self.prepare(**kwargs)
        merge_all_csvs(args)
        elasped = time.time() - start
        elasped = str(datetime.timedelta(seconds=elasped))
        print('done. total time elasped: {}'.format(elasped))

    def reduce_pkls(self, **kwargs):
        start = time.time()
        args = self.prepare(**kwargs)
        reduce_all_pkls(args)
        elasped = time.time() - start
        elasped = str(datetime.timedelta(seconds=elasped))
        print('done. total time elasped: {}'.format(elasped))

    def reduce(self, **kwargs):
        start = time.time()
        args = self.prepare(**kwargs)
        if args.save_cache_as_csvs:
            merge_all_csvs(args)
        else:
            reduce_all_pkls(args)
        elasped = time.time() - start
        elasped = str(datetime.timedelta(seconds=elasped))
        print('done. total time elasped: {}'.format(elasped))

    def compare_measures(self, **kwargs):
        args = self.prepare(**kwargs)
        compare_measures(args)
        print('done')

    def merge_contrastive(self, **kwargs):
        args = self.prepare(**kwargs)
        merge_contrastive(args)


def run(args):
    if args.measure_name == 'contrastive':
        if args.chunk_size is None:
            run_single_contrastive(args)
        else:
            run_chunks_contrastive(args)
    else:
        if args.chunk_size is None:
            run_single(args)
        else:
            run_chunks(args)


if __name__ == '__main__':
    fire.Fire(Cli)
