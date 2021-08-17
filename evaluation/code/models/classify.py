import math

import torch
import torch.nn as nn

from models.head_helper import ResNetBasicHead as VisualHead
from models.audio_head_helper import ResNetBasicHead as AudioHead
from models.video_model_builder import _POOL1
from utils.weight_init_helper import init_weights
from models.build import MODEL_REGISTRY


def get_visual_dim_in(
    trans_func,
    width_per_group,
    pooling,
):
    """
    Compute the input dimension to the VisualClassifyHead.
    args:
        trans_func (string): transform function to be used to contrusct each
            ResBlock.
        width_per_group (int): width of each group.
        pooling (bool): Whether to use the output of the ConvNet.
    returns:
        dim_in (int or list): the input dimension.
    """
    if trans_func == "basic_transform":
        factor = 2 ** 3
    elif trans_func == "bottleneck_transform":
        factor = 4 * (2 ** 3)
    else:
        raise NotImplementedError(
            "Does not support {} transfomration".format(trans_func)
        )
    dim_in = [width_per_group * factor]

    if pooling:
        dim_in = sum(dim_in)

    return dim_in


def get_audio_dim_in(
    trans_func,
    width_per_group,
):
    """
    Compute the input dimension to the AudioClassifyHead.
    args:
        trans_func (string): transform function to be used to contrusct each
            ResBlock.
        width_per_group (int): width of each group.
    returns:
        dim_in (int): the input dimension.
    """
    if trans_func == "basic_transform":
        factor = 2 ** 3
    elif trans_func == "bottleneck_transform":
        factor = 4 * (2 ** 3)
    else:
        raise NotImplementedError(
            "Does not support {} transfomration".format(trans_func)
        )
    dim_in = width_per_group * factor
    return dim_in


def get_visual_pool_size(
    vis_arch,
    num_frames,
    crop_size,
):
    """
    Compute the pooling size used in VisualClassifyHead.
    args:
        vis_arch (string): the architecture of the visual conv net.
        num_frames (int): number of frames per clip.
        crop_size (int): spatial size of frames.
    returns:
        pool_size (list): list of p the kernel sizes of spatial tempoeral
            poolings, temporal pool kernel size, height pool kernel size,
            width pool kernel size in order.
    """
    _pool_size = _POOL1[vis_arch]
    _num_frames = num_frames // 2
    pool_size = [
        [
            _num_frames // _pool_size[0][0],
            math.ceil(crop_size / 32) // _pool_size[0][1],
            math.ceil(crop_size / 32) // _pool_size[0][2],
        ]
    ]

    return pool_size


def get_audio_pool_size(
    audio_frequency,
    audio_time,
):
    """
    Compute the pooling size used in AudioClassifyHead.
    args:
        audio_frequency (int): frequency dimension of the audio clip.
        audio_time (int): time dimension of the audio clip.
    returns:
        pool_size (list): list of the kernel sizes of an avg pooling,
            frequency pool kernel size, time pool kernel size in order.
    """
    pool_size = [
        math.ceil(audio_frequency / 16),
        math.ceil(audio_time / 16),
    ]

    return pool_size


class MultimodalHead(nn.Module):
    """
    Multimodal head for the conv net outputs.
    This layer concatenate the outputs of audio and visual convoluational nets
    and performs a fully-connected projection
    """
    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the visual/audio inputs.
            num_classes (int): the channel dimension of the output.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(MultimodalHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        if act_func == "softmax":
            self.act = nn.Softmax(dim=-1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x, y):
        xy_cat = torch.cat((x, y), dim=-1)
        if hasattr(self, "dropout"):
            xy_cat = self.dropout(xy_cat)
        xy_cat = self.projection(xy_cat)
        if not self.training:
            xy_cat = self.act(xy_cat)
        return xy_cat


class ClassifyHead(nn.Module):
    """
    Classification head.
    For linear evaluation, only this classification head will be trained.
    """
    def __init__(
        self,
        cfg,
    ):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ClassifyHead, self).__init__()
        self.cfg = cfg
        if cfg.MODEL.TASK in ["VisualClassify"]:
            visual_dim_in = get_visual_dim_in(
                cfg.RESNET.TRANS_FUNC,
                cfg.RESNET.WIDTH_PER_GROUP,
                False,
            )
            visual_pool_size = get_visual_pool_size(
                cfg.VIS.ARCH,
                cfg.DATA.NUM_FRAMES,
                cfg.DATA.CROP_SIZE,
            )
            self.visual_head = VisualHead(
                dim_in=visual_dim_in,
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=visual_pool_size,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
            )
        elif cfg.MODEL.TASK in ["AudioClassify"]:
            audio_dim_in = get_audio_dim_in(
                cfg.AUDIO_RESNET.TRANS_FUNC,
                cfg.AUDIO_RESNET.WIDTH_PER_GROUP,
            )
            audio_pool_size = get_audio_pool_size(
                cfg.DATA.AUDIO_FREQUENCY,
                cfg.DATA.AUDIO_TIME,
            )
            self.audio_head = AudioHead(
                dim_in=audio_dim_in,
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=audio_pool_size,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
            )
        elif cfg.MODEL.TASK in ["MultimodalClassify"]:
            visual_dim_in = get_visual_dim_in(
                cfg.RESNET.TRANS_FUNC,
                cfg.RESNET.WIDTH_PER_GROUP,
                True,
            )
            audio_dim_in = get_audio_dim_in(
                cfg.AUDIO_RESNET.TRANS_FUNC,
                cfg.AUDIO_RESNET.WIDTH_PER_GROUP,
            )
            self.multimodal_head = MultimodalHead(
                dim_in=(visual_dim_in, audio_dim_in),
                num_classes=cfg.MODEL.NUM_CLASSES,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
            )

        init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.MODEL.ZERO_INIT_FINAL_BN,
        )

    def visual_forward(self, visual_feature_map):
        return self.visual_head(visual_feature_map)

    def audio_forward(self, audio_feature_map):
        return self.audio_head(audio_feature_map)

    def multimodal_forward(self, visual_feature, audio_feature):
        return self.multimodal_head(visual_feature, audio_feature)

    def forward(self, feature_maps):
        if self.cfg.MODEL.TASK in ["VisualClassify"]:
            return self.visual_forward(*feature_maps)
        elif self.cfg.MODEL.TASK in ["AudioClassify"]:
            return self.audio_forward(*feature_maps)
        elif self.cfg.MODEL.TASK in ["MultimodalClassify"]:
            return self.multimodal_forward(*feature_maps)


@MODEL_REGISTRY.register()
class VisualClassify(nn.Module):
    """
    Visual classifier
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(VisualClassify, self).__init__()
        self.cfg = cfg
        self.visual_conv = MODEL_REGISTRY.get(cfg.VIS.MODEL_NAME)(cfg)
        init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.MODEL.ZERO_INIT_FINAL_BN,
        )

    def forward(self, visual_clip):
        return (self.visual_conv.get_feature_map(visual_clip), )


@MODEL_REGISTRY.register()
class AudioClassify(nn.Module):
    """
    Audio classifier
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(AudioClassify, self).__init__()
        self.cfg = cfg
        self.audio_conv = MODEL_REGISTRY.get(cfg.AUD.MODEL_NAME)(cfg)
        init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.MODEL.ZERO_INIT_FINAL_BN,
        )

    def forward(self, audio_clip):
        return (self.audio_conv.get_feature_map(audio_clip), )


@MODEL_REGISTRY.register()
class MultimodalClassify(nn.Module):
    """
    Multimodal classifier
    """
    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(MultimodalClassify, self).__init__()
        self.cfg = cfg
        self.visual_conv = MODEL_REGISTRY.get(cfg.VIS.MODEL_NAME)(cfg)
        self.audio_conv = MODEL_REGISTRY.get(cfg.AUD.MODEL_NAME)(cfg)
        init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.MODEL.ZERO_INIT_FINAL_BN,
        )

    def forward(self, visual_clip, audio_clip):
        return (self.visual_conv(visual_clip), self.audio_conv(audio_clip))
