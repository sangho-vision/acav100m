import fire

from args import get_args
from script import parallel_extraction_script
from save import store_shards_set


class Cli:
    def run(self, **kwargs):
        return self.cluster(**kwargs)

    def cluster(self, **kwargs):
        if 'out_path' in kwargs:
            kwargs['data.output.path'] = kwargs.pop('out_path')
        if 'feature_path' in kwargs:
            kwargs['shards_path'] = kwargs.pop('feature_path')
        if 'shards_path' in kwargs:
            kwargs['data.path'] = kwargs.pop('shards_path')
        if 'meta_path' in kwargs:
            kwargs['data.meta.path'] = kwargs.pop('meta_path')

        args = get_args(**kwargs)
        args.data.output.path.mkdir(parents=True, exist_ok=True)

        saved_paths = parallel_extraction_script(args)
        store_shards_set(args, saved_paths)
        print('done')


if __name__ == '__main__':
    fire.Fire(Cli)
