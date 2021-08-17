from pathlib import Path

import fire

from args import get_args
from models import get_model_dict
from script import parallel_extraction_script


class Cli:
    def run(self, **kwargs):
        return self.extract(**kwargs)

    def extract(self, **kwargs):
        if 'out_path' in kwargs:
            kwargs['data.output.path'] = kwargs.pop('out_path')
        if 'tar_path' in kwargs:
            kwargs['shards_path'] = kwargs.pop('tar_path')
        if 'shards_path' in kwargs:
            kwargs['data.path'] = kwargs.pop('shards_path')
        if 'meta_path' in kwargs:
            kwargs['data.meta.path'] = kwargs.pop('meta_path')

        args = get_args(**kwargs)
        if args.acav.model_cache_path is not None:
            args.data.cache_dir = Path(args.acav.model_cache_path) / 'cache'
            args.data.cache_dir.mkdir(exist_ok=True, parents=True)
        args.data.output.path.mkdir(parents=True, exist_ok=True)

        parallel_extraction_script(args)
        print('done')

    def show_model_dict(self):
        print(get_model_dict())


if __name__ == '__main__':
    fire.Fire(Cli)
