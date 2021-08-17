"""Visual Conv models."""

import torch
import torch.nn as nn
import math

from models import head_helper, resnet_helper, stem_helper
from models.build import MODEL_REGISTRY

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 34: (3, 4, 6, 3), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "resnet": [
        [[5]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
}

_POOL1 = {
    "resnet": [[1, 1, 1]],
}


@MODEL_REGISTRY.register()
class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.num_pathways = 1
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
            dim_in (list): list containing the input dimension.
        """
        if trans_func == 'basic_transform':
            factor = 1 if idx == 0 else 2 ** (idx - 1)
        elif trans_func == 'bottleneck_transform':
            factor = 1 if idx == 0 else 2 * (2 ** idx)
        else:
            raise NotImplementedError(
                "Does not support {} transfomration".format(trans_func)
            )

        dim_in = [width_per_group * factor]
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
            dim_out (list): list containing the output dimension.
        """
        if trans_func == 'basic_transform':
            factor = 2 ** idx
        elif trans_func == 'bottleneck_transform':
            factor = 4 * (2 ** idx)
        else:
            raise NotImplementedError(
                "Does not support {} transfomration".format(trans_func)
            )

        dim_out = [width_per_group * factor]
        return dim_out

    def _construct_network(self, cfg):
        """
        Builds a ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.VIS.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.VIS.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.VIS.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[2, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
        )

        dim_in_l = [
            self._compute_dim_in(
                i,
                cfg.RESNET.TRANS_FUNC,
                width_per_group
            )
            for i in range(4)
        ]

        dim_out_l = [
            self._compute_dim_out(
                i,
                cfg.RESNET.TRANS_FUNC,
                width_per_group
            )
            for i in range(4)
        ]

        self.s2 = resnet_helper.ResStage(
            dim_in=dim_in_l[0],
            dim_out=dim_out_l[0],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=dim_in_l[1],
            dim_out=dim_out_l[1],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=dim_in_l[2],
            dim_out=dim_out_l[2],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=dim_in_l[3],
            dim_out=dim_out_l[3],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
        )

        _num_frames = cfg.DATA.NUM_FRAMES // 2

        self.head = head_helper.ResNetPoolingHead(
            dim_in=dim_out_l[3],
            pool_size=[
                [
                    _num_frames // pool_size[0][0],
                    math.ceil(cfg.DATA.CROP_SIZE / 32) // pool_size[0][1],
                    math.ceil(cfg.DATA.CROP_SIZE / 32) // pool_size[0][2],
                ]
            ],
        )

        self.output_size = sum(dim_out_l[3])

    def get_feature_map(self, x):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        return x

    def get_logit(self, feature_map):
        return self.head(feature_map)

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.head(x)
        return x
