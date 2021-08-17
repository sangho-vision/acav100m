import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.distributed as du
from utils.weight_init_helper import init_weights
from utils.metrics import topk_accuracies, topks_correct
from models.build import MODEL_REGISTRY
from models.utils import FFNLayer


@MODEL_REGISTRY.register()
class Contrast(nn.Module):
    """
    Pretraining model with the cross-modal (audio-visual) contrastive task
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(Contrast, self).__init__()
        self.cfg = cfg
        self.visual_conv = MODEL_REGISTRY.get(cfg.VIS.MODEL_NAME)(cfg)
        self.visual_mlp = FFNLayer(
            self.visual_conv.output_size,
            self.visual_conv.output_size,
            cfg.CONTRAST.PROJECTION_SIZE,
            activation="relu",
            norm="batch_norm",
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
        )
        self.audio_conv = MODEL_REGISTRY.get(cfg.AUD.MODEL_NAME)(cfg)
        self.audio_mlp = FFNLayer(
            self.audio_conv.output_size,
            self.audio_conv.output_size,
            cfg.CONTRAST.PROJECTION_SIZE,
            activation='relu',
            norm="batch_norm",
            eps=cfg.MODEL.EPSILON,
            bn_mmt=cfg.MODEL.MOMENTUM,
        )

        init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.MODEL.ZERO_INIT_FINAL_BN,
        )

    @property
    def has_to_gather(self):
        """
        If CONTRAST.USE_GLOBAL_BATCH is True, we gather representations across
        multiple gpus
        """
        return du.get_world_size() > 1 and self.cfg.CONTRAST.USE_GLOBAL_BATCH

    def get_conv_output(self, visual_clip, audio_clip):
        """
        Feed audio/visual clip into audio/visul cnn.
        Args:
            visual_clip (tensor): the batch of visual frames. The dimension is
                `batch_size ` x `channel` x `num_frames` x ``height` x `width`.
            audio_clip (tensor): the batch of log-mel-spectrograms. The
                dimension is `batch_size` x `channel` x `frequency` x `time`.
        Returns:
            visual_conv_output (tensor): visual cnn representations.
                `batch_size` x `visual_conv.output_size`
            audio_conv_output (tensor): audio cnn representations.
                `batch_size` x `audio_conv.output_size`
        """
        visual_conv_output = self.visual_conv(visual_clip)
        audio_conv_output = self.audio_conv(audio_clip)

        return visual_conv_output, audio_conv_output

    def forward(self, visual_clip, audio_clip):
        # visual/audio cnn representations
        visual_conv_output, audio_conv_output = self.get_conv_output(
            visual_clip, audio_clip,
        )

        batch_size = visual_conv_output.size(0)

        # Feed into mlp projection heads and l2 normalize
        out1 = F.normalize(self.visual_mlp(visual_conv_output), dim=-1)
        out2 = F.normalize(self.audio_mlp(audio_conv_output), dim=-1)

        # If CONTRAST.USE_GLOBAL_BATCH is True, gather representations across
        # multiple gpus
        # gb_size = batch_size x # of gpus
        gb_size = batch_size
        if self.has_to_gather:
            gb_size = batch_size * du.get_world_size()
            out1_large = du.diff_all_gather(out1)
            out2_large = du.diff_all_gather(out2)
            # Adjust labels of contrastive task considering rank of the current
            # node
            labels = (
                torch.arange(batch_size, device=out1.device)
                + du.get_rank() * batch_size
            )
        else:
            out1_large = out1
            out2_large = out2
            labels = torch.arange(batch_size, device=out1.device)

        # Cross-modal contrastive task for the current node
        # visual_query: out1 -> audio_query: out2_large (gathered across gpus)
        # audio_query: out2 -> visual_query: out1_large (gathered across gpus)
        logits_ab = (
            out1.matmul(out2_large.transpose(0, 1))
        ) / self.cfg.CONTRAST.TEMPERATURE

        logits_ba = (
            out2.matmul(out1_large.transpose(0, 1))
        ) / self.cfg.CONTRAST.TEMPERATURE


        loss_a = F.cross_entropy(
            logits_ab,
            labels,
            reduction='sum',
        )
        loss_b = F.cross_entropy(
            logits_ba,
            labels,
            reduction='sum',
        )

        loss = loss_a + loss_b
        loss = loss / (2 * batch_size)

        corrects_a = topks_correct(
            logits_ab,
            labels,
            [1],
        )[0]
        corrects_b = topks_correct(
            logits_ba,
            labels,
            [1],
        )[0]
        acc = (corrects_a + corrects_b) / (2 * batch_size) * 100.0

        return loss, acc
