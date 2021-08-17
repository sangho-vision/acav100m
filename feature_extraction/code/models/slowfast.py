from functools import partial

import torch
from torch import nn

from .utils_slowfast import (
    set_cache_dir,
    load_model_options,
    load_model,
    preprocess as _preprocess
)


def preprocess(visual, audio, cfg):
    frames, fps = visual
    frames = _preprocess(cfg, frames)
    return {'data': frames, 'fps': fps}


'''
model_tags = {
    'Kinetics/c2/SLOWFAST_8x8_R50':
        {
            'name': 'SLOWFAST_8x8_R50',
            'dataset': 'kinetics-400',
        }
}
'''


class SlowFast(nn.Module):
    args = {
        'slowfast_config': 'Kinetics/c2/SLOWFAST_8x8_R50'
    }
    model_tag = {
        'name': 'SLOWFAST_8x8_R50',
        'dataset': 'kinetics-400',
    }
    output_dims = 2304

    def __init__(self, args):
        super().__init__()

        # self.model_tag = self.model_tags[args.slowfast_config]
        self.cache_dir = set_cache_dir(args.data.cache_dir)
        self.model_choices = load_model_options(self.cache_dir)
        self.model, self.cfg = load_model(args.slowfast_config, self.model_choices,
                                          self.cache_dir, args.computation.device)

    @classmethod
    def download(cls, args):
        cache_dir = set_cache_dir(args.data.cache_dir)
        model_choices = load_model_options(cache_dir)
        model, cfg = load_model(args.slowfast_config, model_choices,
                                cache_dir, args.computation.device)
        return model

    def get_preprocessor(self):
        return partial(preprocess, cfg=self.cfg)

    def _forward(self, x):
        model = self.model
        x = model.s1(x)
        x = model.s1_fuse(x)
        x = model.s2(x)
        x = model.s2_fuse(x)
        for pathway in range(model.num_pathways):
            pool = getattr(model, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = model.s3(x)
        x = model.s3_fuse(x)
        x = model.s4(x)
        x = model.s4_fuse(x)
        x = model.s5(x)

        head = self.model.head
        assert (
            len(x) == head.num_pathways
        ), "Input tensor does not contain {} pathway".format(head.num_pathways)
        pool_out = []
        for pathway in range(head.num_pathways):
            m = getattr(head, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(x[pathway]))
        x = torch.cat(pool_out, 1)
        # (B, C, T, H, W) -> (B, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        return x

    def forward(self, data):
        x = data
        # BTHWC: BatchSize, NumFrames, Height, Width, Channels
        x = self._forward(x)
        x = x.mean([1, 2, 3])
        # BC
        return x


class LayerSlowFast(SlowFast):
    args = {
        'slowfast_config': 'Kinetics/c2/SLOWFAST_8x8_R50',
        'num_layers': 5
    }
    output_dims = [88, 352, 704, 1408, 2304]

    def __init__(self, args):
        super().__init__(args)
        self.num_layers = args.num_layers

    def _forward(self, x):
        model = self.model
        xs = []
        x = model.s1(x)
        x = model.s1_fuse(x)
        xs.append([v.clone().detach() for v in x])
        x = model.s2(x)
        x = model.s2_fuse(x)
        for pathway in range(model.num_pathways):
            pool = getattr(model, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        xs.append([v.clone().detach() for v in x])
        x = model.s3(x)
        x = model.s3_fuse(x)
        xs.append([v.clone().detach() for v in x])
        x = model.s4(x)
        x = model.s4_fuse(x)
        xs.append([v.clone().detach() for v in x])
        x = model.s5(x)
        xs.append([v.clone().detach() for v in x])

        head = self.model.head
        assert (
            len(x) == head.num_pathways
        ), "Input tensor does not contain {} pathway".format(head.num_pathways)

        def get_pool(x):
            pool_out = []
            for pathway in range(head.num_pathways):
                m = getattr(head, "pathway{}_avgpool".format(pathway))
                pool_out.append(m(x[pathway]))
            x = torch.cat(pool_out, 1)
            # (B, C, T, H, W) -> (B, T, H, W, C).
            x = x.permute((0, 2, 3, 4, 1))
            return x

        xs = [get_pool(x) for x in xs]
        return xs

    def forward(self, data, no_grad=True):
        x = data
        # BTHWC: BatchSize, NumFrames, Height, Width, Channels
        if no_grad:
            for i, _ in enumerate(x):
                x[i].requires_grad_(False)
        xs = self._forward(x)
        xs = [x.mean([1, 2, 3]) for x in xs]
        # BC
        return xs
