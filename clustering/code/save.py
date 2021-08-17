from pathlib import Path
from collections import OrderedDict

from tqdm import tqdm

from utils import dump_pickle, dump_json, load_pickle, get_idx


def store_shards_set(args, saved_paths):
    if len(saved_paths) == 0:
        print("All shards already processed")
        return
    saved_names = [p.stem for p in saved_paths]
    out_dir = saved_paths[0].parent
    out_path = out_dir / ('log_' + args.run_id + '.json')
    res = {**args.run_info, 'shards': saved_names}
    dump_json(res, out_path, indent=None)


def save_cache(*args, **kwargs):
    return save_output(*args, **kwargs, suffix='_cache.pkl')


def remove_cache(args, shard_name):
    cache_path = args.data.output.path / (shard_name + '_cache.pkl')
    if cache_path.is_file():
        cache_path.unlink()


def save_output(args, models, shards, ids, shard_name,
                name='features', suffix='.pkl', final=False, prefix=''):
    if final:
        remove_cache(args, shard_name)
    data = []
    for model_name, model in models.items():
        if args.computation.device == 'cuda' and hasattr(model, 'module'):
            _model = model.module
        else:
            _model = model
        datum = {'model_key': model_name, 'data': shards[model_name][shard_name],
                    **_model.model_tag}
        data.append(datum)

    return _save_output(args, shard_name, ids[shard_name], data, name=name,
                        suffix=suffix, prefix=prefix)


def _save_output(args, shard_name, ids, data, name='features', suffix='.pkl', prefix=''):
    res = []
    for idx in ids:
        row = {}
        row['video_{}'.format(name)] = []
        row['audio_{}'.format(name)] = []
        for model_feat in data:
            feature = {}
            feature['model_key'] = model_feat['model_key']
            feature['extractor_name'] = model_feat['name']
            feature['dataset'] = model_feat['dataset']  # pretrained dataset
            point_feat = model_feat['data'][idx]
            feature['array'] = point_feat[name]
            if isinstance(feature['array'], (tuple, list)):  # layer extractor
                feature['array'] = {"layer_{}".format(i): v for i, v in enumerate(feature['array'])}
            meta_keys = ['filename', 'shard_size', 'shard_name']
            for key in meta_keys:
                row[key] = point_feat[key]
            model_name = model_feat['model_key']
            if model_name in args.model_types.audio:
                row['audio_{}'.format(name)].append(feature)
            else:
                row['video_{}'.format(name)].append(feature)
        res.append(row)
    out_path = get_out_path(args, prefix + shard_name + suffix)
    dump_pickle(res, out_path)
    return out_path


def get_out_path(args, name):
    out_path = args.data.output.path / name
    out_path.parent.mkdir(exist_ok=True, parents=True)
    return out_path


def load_cache_features(caches, model_name, shard_name):
    # reverse save_output
    cache = caches[shard_name]
    res = {}
    meta_keys = ['filename', 'shard_size', 'shard_name']
    for row in cache:
        features = [*row['audio_features'], *row['video_features']]
        for feature in features:
            if feature['model_key'] == model_name:
                idx = get_idx(row['filename'])
                if isinstance(feature['array'], dict):  # layer extractor
                    layer_keys = sorted(feature['array'].keys())
                    feature['array'] = [feature['array'][key] for key in layer_keys]
                row_out = {'features': feature['array']}
                for meta_key in meta_keys:
                    row_out[meta_key] = row[meta_key]
                res[idx] = row_out
        try:
            temp = res[idx]
        except Exception as e:
            print("feature ndot present error in shard: {}".format(shard_name))
            print(e)
    return res


def get_processed_paths(args, paths):
    shard_names = [p.stem for p in paths]
    if args.clustering.cached_epoch is not None:
        epoch_name = 'epoch_{}_'.format(args.clustering.cached_epoch)
        out_paths = [args.data.output.path / (epoch_name + shard_name + '.pkl') for shard_name in shard_names]
    else:
        out_paths = [args.data.output.path / (shard_name + '.pkl') for shard_name in shard_names]
    out_paths = [p for p in out_paths if p.is_file()]
    return out_paths


def load_shard_caches(args, shards_path):
    print("loading shard caches")
    caches = {}
    skip_lists = OrderedDict()
    for shard_path in tqdm(shards_path):
        shard_name = Path(shard_path).stem
        cache_path = args.data.output.path / (shard_name + '_cache.pkl')
        if cache_path.is_file():
            cache = load_pickle(cache_path)
            skip_list = [row['filename'] for row in cache]
            caches[shard_name] = cache
            skip_lists[shard_name] = skip_list
        else:
            skip_lists[shard_name] = []
    return caches, skip_lists
