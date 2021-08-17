import math
from collections import defaultdict

from tqdm import tqdm, trange

import torch
import numpy as np

from utils import peek
from .pca import PCAOptim
from .contrastive import Contrastive
from .dataloaders import (
    ClassDataLoader, SampleDataLoader, InferDataLoader,
    get_penultimates
)


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


class MetricLearning(PCAOptim):
    def __init__(self, args, data):
        train, test = data
        self.train_data = train
        self.test_data = test
        self.num_epochs = args.num_epochs
        self.batch_size = args.contrastive_batch_size  # not applicable for class-level training
        self.sample_level = args.sample_level
        self.device = args.device
        self.use_test_for_train = args.use_test_for_train
        self.base_lr = args.base_lr

        sizes = self.get_sizes(train)
        self.model = Contrastive(*sizes)
        self.model = self.model.to(self.device)

    def init(self, clustering_combinations, candidates):
        pass

    def get_sizes(self, train):
        class_data = peek(train)
        row = class_data[0]
        penultimates = get_penultimates(list(row['features'].keys()))
        return [row['features'][k].shape[-1] for k in penultimates]

    def get_feature_names(self, train):
        class_data = peek(train)
        row = peek(class_data)
        return sorted(list(row.keys()))

    def get_train_dataloader(self):
        '''
        if self.use_test_for_train:
            data = self.test_data
            return SampleDataLoader(data, self.batch_size)
        else:
            '''
        data = self.train_data
        if self.sample_level:
            return SampleDataLoader(data, self.batch_size)
        else:
            return ClassDataLoader(data)

    def get_infer_dataloader(self):
        data = self.test_data
        return InferDataLoader(data, self.batch_size)

    def train_batch(self, batch, optimizer):
        moved = []
        for feature in batch:
            moved.append(feature.to(self.device))
        loss, acc = self.model(*moved)
        loss.backward()
        optimizer.step()
        return loss.item(), acc.item()

    def train(self):
        print("begin metric training")
        self.model.train()
        optimizer = get_optimizer(self.model.parameters(), self.base_lr)
        for epoch in trange(self.num_epochs, desc='epoch'):
            optimizer, lr = update_lr(optimizer, epoch, self.num_epochs, self.base_lr)
            epoch_loss = []
            epoch_acc = []
            dataloader = self.get_train_dataloader()  # shuffle
            pbar = tqdm(dataloader, total=len(dataloader), desc="iters")
            for batch in pbar:
                loss, acc = self.train_batch(batch, optimizer)
                epoch_loss.append(loss)
                epoch_acc.append(acc)
                pbar.set_description("iters (lr: {:04f}, loss: {:04f}, acc: {:04f})".format(lr, loss, acc))
            epoch_loss = np.array(epoch_loss).mean()
            epoch_acc = np.array(epoch_acc).mean()
            tqdm.write("epoch {} done (lr: {:04f}, loss: {:04f}, acc: {:04f})".format(epoch, lr, epoch_loss, epoch_acc))
        print("finished metric training")
        return

    def infer_batch(self, batch):
        moved = []
        for feature in batch:
            moved.append(feature.to(self.device))
        logits = self.model.infer(*moved)
        return logits.detach().cpu()

    def infer(self):
        with torch.no_grad():
            res = self._infer()
        return res

    def _infer(self):
        print("infering metrics distances")
        logits = []
        dataloader = self.get_infer_dataloader()
        pbar = tqdm(dataloader, total=len(dataloader), desc="iters")
        for batch in pbar:
            logits.append(self.infer_batch(batch))
        logits = torch.cat(logits, dim=0)
        print("finished infering metrics distances")
        return logits

    def calc_measure(self, subset_size):
        self.train()
        logits = self.infer()  # V
        scores, ids = logits.topk(subset_size, sorted=True)
        return scores, ids
