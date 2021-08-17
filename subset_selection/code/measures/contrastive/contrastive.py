import csv
from pathlib import Path

import torch
import pandas
import numpy as np

from utils import peek, load_json, dump_json
from .module import ContrastiveModule
from mps import distributed as du
from save import format_rows


def get_penultimates(keys):
    penultimates = {}
    for key in keys:
        view = key[:key.find('_')]  # get dataset+model name
        layer_name = key[key.find('_') + 1:]
        if view not in penultimates:
            penultimates[view] = view + '_' + layer_name
        elif layer_name > penultimates[view]:
            penultimates[view] = view + '_' + layer_name
    keys = sorted(list(penultimates.keys()))
    return [penultimates[k] for k in keys]


def get_optimizer(params, lr=1e-3):
    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-6,
        amsgrad=True,
    )
    return optimizer


def set_lr(optimizer, lr):
    for param in optimizer.param_groups:
        param['lr'] = lr
    return optimizer


def lr_func_linear(current_step, num_training_steps, num_warmup_steps=3):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def update_lr(optimizer, epoch, num_epochs, base_lr=1e-3, num_warmup_steps=3):
    lr = lr_func_linear(epoch + 1, num_epochs + 1, num_warmup_steps) * base_lr
    optimizer = set_lr(optimizer, lr)
    return optimizer, lr


class Contrastive:
    def __init__(self, num_epochs=1, device='cpu', base_lr=1e-4,
                 num_warmup_steps=3, distributed=False):
        self.num_epochs = num_epochs
        self.device = device
        self.base_lr = base_lr
        self.num_warmup_steps = num_warmup_steps
        self.distributed = distributed
        self.epoch = 0

        # sizes = self.get_sizes(train)
        sizes = self.default_sizes
        self.model = ContrastiveModule(*sizes, use_global_batch=distributed)
        self.model = self.model.to(self.device)

    def init(self, clustering_combinations, candidates):
        pass

    @property
    def default_sizes(self):
        # video (slowfast) : 2304, audio (VGGish) : 128
        return [2304, 128]

    def get_sizes(self, train):
        class_data = peek(train)
        row = class_data[0]
        penultimates = get_penultimates(list(row['features'].keys()))
        return [row['features'][k].shape[-1] for k in penultimates]

    def get_feature_names(self, train):
        class_data = peek(train)
        row = peek(class_data)
        return sorted(list(row.keys()))

    def train_batch(self, batch, optimizer):
        moved = []
        for feature in batch:
            moved.append(feature.to(self.device))
        loss, acc = self.model(*moved)
        loss.backward()
        if self.distributed:
            self.model.average_gradient()
        optimizer.step()
        return loss.item(), acc.item()

    def _get_features(self, batch):
        unique_ids = pandas.Series(batch['idx']).drop_duplicates().index.tolist()
        filenames = [batch['filename'][idx] for idx in unique_ids]
        ids = [batch['idx'][idx] for idx in unique_ids]
        shard_names = [batch['shard_name'][idx] for idx in unique_ids]
        metas = [{'id': idx, 'filename': filename, 'shard_name': shard_name}
                  for idx, filename, shard_name in zip(ids, filenames, shard_names)]

        video_features = batch['SLOWFAST_8x8_R50/kinetics-400']['layer_4']
        audio_features = batch['VGGish/YouTube-8M']['layer_4']
        unique_ids = torch.Tensor(unique_ids).long()
        video_features = video_features.index_select(dim=0, index=unique_ids)
        audio_features = audio_features.index_select(dim=0, index=unique_ids)
        return metas, [video_features, audio_features]

    def get_features(self, batch):
        metas, [video_features, audio_features] = self._get_features(batch)
        if self.distributed:
            i = du.get_rank()
            total = du.get_world_size()
            metas = metas[i::total]
            video_features = video_features[i::total]
            audio_features = audio_features[i::total]
        return metas, [video_features, audio_features]

    def train(self, args, path, dataloader, log_every=1, verbose=True):
        self.model.train()
        optimizer = get_optimizer(self.model.parameters(), self.base_lr)
        for epoch in range(self.epoch, self.num_epochs):
            optimizer, lr = update_lr(optimizer, epoch, self.num_epochs, self.base_lr,
                                      self.num_warmup_steps)
            epoch_loss = []
            epoch_acc = []
            pbar = dataloader
            for count, batch in enumerate(pbar):
                _, features = self.get_features(batch)
                loss, acc = self.train_batch(features, optimizer)
                epoch_loss.append(loss)
                epoch_acc.append(acc)
                if verbose and count % log_every == 0:
                    print("(node {}) training epoch ({}/{}) iter ({}/{}) (lr: {:04f}, loss: {:04f}, acc: {:04f})".format(
                        du.get_rank(), epoch, self.num_epochs, count, len(dataloader), lr, loss, acc))
            epoch_loss = np.array(epoch_loss).mean()
            epoch_acc = np.array(epoch_acc).mean()
            if verbose:
                print("(node {}) epoch ({}/{}) done (lr: {:04f}, loss: {:04f}, acc: {:04f})".format(
                    du.get_rank(), epoch, self.num_epochs, lr, epoch_loss, epoch_acc))
            self.epoch = epoch
            self.save_cache(args, path, epoch, verbose)

        return

    def get_cache_path_run(self, args, epoch):
        cache_dir = args.data.output.path.parent / 'caches'
        cache_dir.mkdir(parents=True, exist_ok=True)
        pid = args.parent_pid
        rank = args.node_rank
        i = args.chunk_num
        name = "contrastive_model_cache_epoch_{}_{}_{}_{}.pkl".format(epoch, pid, rank, i)
        path = str(cache_dir / name)
        key_name = "contrastive_model_cache_epoch_{}_{}_{}_{}.json".format(epoch, pid, rank, i)
        key_path = str(cache_dir / key_name)
        return path, key_path

    def get_cache_path_load(self, args, path, epoch):
        cache_dir = args.data.output.path.parent / 'caches'
        cache_dir.mkdir(parents=True, exist_ok=True)
        keys = list(cache_dir.glob("contrastive_model_cache_epoch_{}_*.json".format(epoch)))
        if len(keys) == 0:
            return None
        keys = {p.stem: set(load_json(p)) for p in keys}
        path = set([Path(p).stem for p in path])
        intersections = [(k, len(v & path)) for k, v in keys.items() if len(path - v) == 0]
        if len(intersections) == 0:
            return None
        key = max(intersections, key=lambda x: x[1])[0]
        path = cache_dir / key
        path = path.parent / (path.stem + '.pkl')
        return path

    def save_cache(self, args, chunks, epoch, verbose=True):
        path, key_path = self.get_cache_path_run(args, epoch)
        dt = {
            'epoch': self.epoch,
            'base_lr': self.base_lr,
            'model': self.model.state_dict()
        }
        if verbose:
            print("saved cache file: {}".format(Path(path).stem))
        torch.save(dt, path)
        keys = [Path(p).stem for p in chunks]
        dump_json(keys, key_path)

    def load_cache(self, args, path, epoch):
        path = self.get_cache_path_load(args, path, epoch)
        assert path is not None, 'no cache file'
        dt = torch.load(path)
        self.epoch = dt['epoch']
        self.base_lr = dt['base_lr']
        self.model.load_state_dict(dt['model'])

    def infer_batch(self, batch):
        moved = []
        for feature in batch:
            moved.append(feature.to(self.device))
        logits = self.model.infer(*moved)
        return logits.detach().cpu()

    def infer(self, args, dataloader, json_metas, subset_size, log_every=1, verbose=True):
        self.model.eval()
        with torch.no_grad():
            logits, filename_ids = self._infer(args, dataloader, json_metas, log_every, verbose)
        if subset_size > logits.shape[0]:
            subset_size = logits.shape[0]
        scores, ids = logits.topk(subset_size, sorted=True)
        return scores, ids, filename_ids

    def _infer(self, args, dataloader, json_metas, log_every=1, verbose=True):
        logits = []
        pbar = dataloader
        metas = []
        for count, batch in enumerate(pbar):
            batch_metas, features = self.get_features(batch)
            logit = self.infer_batch(features)
            logits.append(logit)
            metas.extend(batch_metas)
            if verbose and count % log_every == 0:
                print("inference iter ({}/{}) saving caches".format(count, len(dataloader)))
                logits = torch.cat(logits, dim=0)
                self.save_inference(args, logits, metas, json_metas)
                logits = []
                metas = []
        if len(metas) > 0:
            logits = torch.cat(logits, dim=0)
            self.save_inference(args, logits, metas, json_metas)
        print("done: inference iter ({}/{}) saving caches".format(count, len(dataloader)))
        return logits, metas

    def save_inference(self, args, logits, metas, json_metas):
        cache_dir = args.data.output.path.parent / 'caches'
        cache_dir.mkdir(parents=True, exist_ok=True)
        pid = args.parent_pid
        local_rank = du.get_rank()
        output_name = Path(args.data.output.path).stem
        name = "{}_contrastive_inferred_cache_{}_{}.csv".format(output_name, pid, local_rank)

        scores = logits.numpy().tolist()
        rows = [{'score': score, **v} for score, v in zip(scores, metas)]
        lines = format_rows(rows, json_metas, sharded_meta=True,
                            headers=['score', 'shard_name', 'filename', 'id', 'segment'])

        print("saving cache to {}".format(cache_dir / name))
        with open(cache_dir / name, 'a+') as f:
            writer = csv.writer(f)
            for line in lines:
                writer.writerow(line)
