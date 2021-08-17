import numpy as np

from utils import to_device, dol_to_lod


def _train_batch(args, features, clusterings):
    if isinstance(features, list):
        features = {'layer_{}'.format(i): feature for i, feature in enumerate(features)}

    if isinstance(features, dict):
        # layer extractor
        distance = [clustering.add(features[key].to(args.computation.device)) \
                    for key, clustering in clusterings.items()]
        distance = np.array(distance).mean()
    else:
        distance = clusterings['model'].add(features.to(args.computation.device))
    return distance


def train_batch(args, model, batch, options, clusterings):
    batch = to_device(batch, args.computation.device)
    data = batch['data']
    # batch processing
    features = model(data)  # BC
    distance = _train_batch(args, features, clusterings)

    return distance


def train_batch_cached(args, batch, clusterings):
    batch = to_device(batch, args.computation.device)
    features = batch
    distance = _train_batch(args, features, clusterings)
    return distance


def _extract_batch(args, batch, features, clusterings):
    if isinstance(features, list):
        features = {'layer_{}'.format(i): feature for i, feature in enumerate(features)}

    if isinstance(features, dict):
        # layer extractor
        features = {key: clustering.calc_best(features[key].to(args.computation.device))[0] \
                    for key, clustering in clusterings.items()}
        features = {key: list(feature.detach().cpu().numpy()) for key, feature in features.items()}
        features = dol_to_lod(features)
    else:
        features = clusterings['model'].calc_best(features.to(args.computation.device))[0]
        features = list(features.detach().cpu().numpy())

    features = [{'assignments': f} for f in features]
    meta_keys = ['filename', 'shard_name', 'shard_size', 'idx']
    metas = {k: batch[k] for k in meta_keys if k in batch}
    metas = dol_to_lod(metas)
    features = [{**meta, **feature} for meta, feature in zip(metas, features)]
    return features


def extract_batch(args, model, batch, options, clusterings):
    batch = to_device(batch, args.computation.device)
    data = batch['data']
    # batch processing
    features = model(data)  # BC
    return _extract_batch(args, batch, features, clusterings)


def extract_batch_cached(args, meta, batch, clusterings):
    batch = to_device(batch, args.computation.device)
    return _extract_batch(args, meta, batch, clusterings)
