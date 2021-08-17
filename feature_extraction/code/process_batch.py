from tqdm import tqdm

import torch

from utils import to_device, dol_to_lod


def extract_batch(args, model, batch, options):
    batch = to_device(batch, args.computation.device)
    data = batch['data']
    if (not options['pathways_packed']) and isinstance(data, list):
        # loop processing
        features = []
        tqdm.write('loop processing')
        for datum in data:
            feature = model(datum.unsqueeze(0))  # BC
            if isinstance(feature, list):
                # layer extractor
                feature = [feat.squeeze(0).detach().cpu().numpy()
                            for feat in feature]
            else:
                feature = feature.squeeze(0).detach().cpu().numpy()
            features.append(feature)
    else:
        # batch processing
        features = model(data)  # BC
        if isinstance(features, list):
            # layer extractor
            features = [list(feature.detach().cpu().numpy()) for feature in features]
            features = list(zip(*features))
        else:
            features = list(features.detach().cpu().numpy())

    if 'label' in batch:
        label = batch['label']
        if torch.is_tensor(label):
            label = label.detach().cpu().numpy()
        features = [{'features': f, 'label': l} for f, l in zip(features, label)]
    else:
        features = [{'features': f} for f in features]

    meta_keys = ['filename', 'shard_name', 'shard_size', 'idx']
    metas = {k: batch[k] for k in meta_keys if k in batch}
    metas = dol_to_lod(metas)
    features = [{**meta, **feature} for meta, feature in zip(metas, features)]
    return features
