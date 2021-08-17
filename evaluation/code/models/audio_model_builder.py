"""Audio models."""

import torch
import torch.nn as nn
import math

from models import audio_head_helper, audio_resnet_helper, audio_stem_helper
from models.build import MODEL_REGISTRY

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 34: (3, 4, 6, 3), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}


@MODEL_REGISTRY.register()
class AudioResNet(nn.Module):
    """
    Audio Resnet model builder.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(AudioResNet, self).__init__()
        self._construct_network(cfg)

    def _compute_dim_in(
        self,
        idx,
        trans_func,
        width_per_group,
    ):
        """
        Compute the input dimension of each convolutional stage.
        args:
            idx (int): the index of the convolutional stage.
            trans_func (string): transform function to be used to contrusct each
                ResBlock.
            width_per_group (int): width of each group.
        returns:
            dim_in (int): the input dimension.
        """
        if trans_func == 'basic_transform':
            factor = 1 if idx == 0 else 2 ** (idx - 1)
        elif trans_func == 'bottleneck_transform':
            factor = 1 if idx == 0 else 2 * (2 ** idx)
        else:
            raise NotImplementedError(
                "Does not support {} transfomration".format(trans_func)
            )

        dim_in = width_per_group * factor
        return dim_in

    def _compute_dim_out(
        self,
        idx,
        trans_func,
        width_per_group,
    ):
        """
        Compute the output dimension of each convolutional stage.
        args:
            idx (int): the index of the convolutional stage.
            trans_func (string): transform function to be used to contrusct each
                ResBlock.
            width_per_group (int): width of each group.
        returns:
            dim_out (int): the output dimension.
        """
        if trans_func == 'basic_transform':
            factor = 2 ** idx
        elif trans_func == 'bottleneck_transform':
            factor = 4 * (2 ** idx)
        else:
            raise NotImplementedError(
                "Does not support {} transfomration".format(trans_func)
            )

        dim_out = width_per_group * factor
        return dim_out

    def _construct_network(self, cfg):
        """
        Builds a ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.AUDIO_RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.AUDIO_RESNET.DEPTH]

        num_groups = cfg.AUDIO_RESNET.NUM_GROUPS
        width_per_group = cfg.AUDIO_RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        self.s1 = audio_stem_helper.AudioModelStem(
            dim_in=1,
            dim_out=width_per_group,
            kernel=[9, 9],
            stride=[1, 1],
            padding=[4, 4],
            separable=True,
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
        )

        dim_in_l = [
            self._compute_dim_in(
                i,
                cfg.AUDIO_RESNET.TRANS_FUNC,
                width_per_group
            )
            for i in range(4)
        ]

        dim_out_l = [
            self._compute_dim_out(
                i,
                cfg.AUDIO_RESNET.TRANS_FUNC,
                width_per_group
            )
            for i in range(4)
        ]

        self.s2 = audio_resnet_helper.ResStage(
            dim_in=dim_in_l[0],
            dim_out=dim_out_l[0],
            dim_inner=dim_inner,
            stride=cfg.AUDIO_RESNET.STRIDES[0],
            num_blocks=d2,
            num_groups=num_groups,
            trans_func_name=cfg.AUDIO_RESNET.TRANS_FUNC,
            stride_1x1=cfg.AUDIO_RESNET.STRIDE_1X1,
            inplace_relu=cfg.AUDIO_RESNET.INPLACE_RELU,
            dilation=cfg.AUDIO_RESNET.DILATIONS[0],
            separable=True,
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
        )

        self.s3 = audio_resnet_helper.ResStage(
            dim_in=dim_in_l[1],
            dim_out=dim_out_l[1],
            dim_inner=dim_inner * 2,
            stride=cfg.AUDIO_RESNET.STRIDES[1],
            num_blocks=d3,
            num_groups=num_groups,
            trans_func_name=cfg.AUDIO_RESNET.TRANS_FUNC,
            stride_1x1=cfg.AUDIO_RESNET.STRIDE_1X1,
            inplace_relu=cfg.AUDIO_RESNET.INPLACE_RELU,
            dilation=cfg.AUDIO_RESNET.DILATIONS[1],
            separable=True,
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
        )

        self.s4 = audio_resnet_helper.ResStage(
            dim_in=dim_in_l[2],
            dim_out=dim_out_l[2],
            dim_inner=dim_inner * 4,
            stride=cfg.AUDIO_RESNET.STRIDES[2],
            num_blocks=d4,
            num_groups=num_groups,
            trans_func_name=cfg.AUDIO_RESNET.TRANS_FUNC,
            stride_1x1=cfg.AUDIO_RESNET.STRIDE_1X1,
            inplace_relu=cfg.AUDIO_RESNET.INPLACE_RELU,
            dilation=cfg.AUDIO_RESNET.DILATIONS[2],
            separable=False,
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
        )

        self.s5 = audio_resnet_helper.ResStage(
            dim_in=dim_in_l[3],
            dim_out=dim_out_l[3],
            dim_inner=dim_inner * 8,
            stride=cfg.AUDIO_RESNET.STRIDES[3],
            num_blocks=d5,
            num_groups=num_groups,
            trans_func_name=cfg.AUDIO_RESNET.TRANS_FUNC,
            stride_1x1=cfg.AUDIO_RESNET.STRIDE_1X1,
            inplace_relu=cfg.AUDIO_RESNET.INPLACE_RELU,
            dilation=cfg.AUDIO_RESNET.DILATIONS[3],
            separable=False,
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
        )

        self.head = audio_head_helper.ResNetPoolingHead(
            pool_size=[
                math.ceil(cfg.DATA.AUDIO_FREQUENCY / 16),
                math.ceil(cfg.DATA.AUDIO_TIME / 16),
            ],
        )

        self.output_size = dim_out_l[3]

    def get_feature_map(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        return x

    def get_logit(self, feature_map):
        return self.head(feature_map)

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.head(x)
        return x
