import subprocess
import shutil
from pathlib import Path

from utils import load_json, load_pickle
from config import defaults
from models.slowfast import LayerSlowFast
from models.vggish import LayerVggish


def prepare(shards_name='samples/shard-{000000..000009}.tar', **additionals):
    cmd = "python cli.py extract"
    in_root = Path('../data/tests')
    out_root = Path('../data/output')
    model_cache_root = Path('../data')
    shards_name = Path(shards_name)
    name = shards_name.parent
    in_path = in_root / shards_name
    out_path = out_root / '{}_output'.format(name)
    paths = {
        'tar_path': in_path,
        'meta_path': in_path.parent,
        'out_path': out_path,
        'ms100m.model_cache_path': model_cache_root
    }
    make_empty_path(paths['out_path'])

    options = paths
    options = {**options, **additionals}
    options = {"--{}".format(k): str(v) for k, v in options.items()}

    options = ' '.join(["{}={}".format(k, v) for k, v in options.items()])
    cmd = cmd + ' ' + options
    return cmd, in_path.parent, out_path


def make_empty_path(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def check(meta_path, out_dir):
    # metas = sorted(list(braceexpand(str(meta_path))))
    metas = sorted(list(meta_path.glob('shard-*.json')))
    for meta_path in metas:
        name = Path(meta_path).stem + '.pkl'
        out_path = out_dir / name
        assert out_path.is_file(), "out file is not present: {}".format(out_path)
        meta = load_json(meta_path)
        out = load_pickle(out_path)
        out = {'{}.mp4'.format(row['filename']): row for row in out}
        total = len(meta)
        generated = len(list([row for row in meta if row['filename'] in out]))
        assert generated >= total * defaults['data']['output']['shard_ok_ratio'], \
            "shard output has insufficient samples: {} ({}/{})".format(meta_path.stem, generated, total)
        check_feature_shapes(out, meta_path.stem)


def check_feature_shapes(out, shard_name):
    vd = LayerSlowFast.output_dims
    ad = LayerVggish.output_dims
    for k, v in out.items():
        kname = v['shard_name']
        assert kname == shard_name, \
            "shard_name mismatch: (in pickle: {}) (in meta: {}) {}".format(kname, shard_name, k)
        va = v['video_features'][0]['array']
        va = [va[k] for k in sorted(list(va.keys()))]
        vas = [v.shape[0] for v in va]
        assert len(vas) == len(vd), \
            "insufficient video feature num: {},{} ({}/{})".format(shard_name, k, vas, vd)
        assert all([var == vdr for var, vdr in zip(vas, vd)]), \
            "wrong video feature dim: {},{} ({}/{})".format(shard_name, k, vas, vd)

        aa = v['audio_features'][0]['array']
        aa = [aa[k] for k in sorted(list(aa.keys()))]
        aas = [a.shape[0] for a in aa]
        assert len(aas) == len(ad), \
            "insufficient audio feature num: {},{} ({}/{})".format(shard_name, k, aas, ad)
        assert all([aar == adr for aar, adr in zip(aas, ad)]), \
            "wrong audio feature dim: {},{} ({}/{})".format(shard_name, k, aas, ad)
        vam = sum([v.mean() ** 2 for v in va])
        assert vam > 0, "all video feature zero?: {},{}".format(shard_name, k)
        aam = sum([a.mean() for a in aa])
        assert aam > 0, "all audio feature zero?: {},{}".format(shard_name, k)


tiny_kwargs = {'computation.num_workers': 0,
               'computation.num_gpus': 1,
               'ms100m.force_cache_restart': True,
               'data.batch_size': 32}


def test_tiny():
    cmd, meta_path, out_path = prepare('samples_tiny/shard-{000000..000003}.tar',
                                       **tiny_kwargs)
    print('running')
    subprocess.run(cmd.split())
    print('testing')
    check(meta_path, out_path)


def test_tiny_range():
    cmd, meta_path, out_path = prepare('samples_tiny_range/shard-{000000..000003}.tar',
                                       **tiny_kwargs)
    print('running')
    subprocess.run(cmd.split())
    print('testing')
    check(meta_path, out_path)


def test_sample():
    cmd, meta_path, out_path = prepare('samples/shard-{000000..000009}.tar')
    print('running')
    subprocess.run(cmd.split())
    print('testing')
    check(meta_path, out_path)


def test_sample_range():
    cmd, meta_path, out_path = prepare('samples_range/shard-{000000..000029}.tar')
    print('running')
    subprocess.run(cmd.split())
    print('testing')
    check(meta_path, out_path)


def test_sample_shards():
    cmd, meta_path, out_path = prepare('shards/shard-{000100..000104}.tar')
    '''
    try:
        subprocess.run(cmd.split())
    except FileNotFoundError as e:
        assert 'shard-000104.json' in e
    '''
    print('running')
    subprocess.run(cmd.split())
    print('testing')
    check(meta_path, out_path)
