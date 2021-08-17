import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
from mps import distributed as du


class ContrastiveModule(nn.Module):
    """
    model with audio-visual contrastive task
    """
    def __init__(self, visual_size, audio_size, out_size=None, use_global_batch=False):
        super().__init__()

        self.use_global_batch = use_global_batch

        if out_size is None:
            out_size = min(visual_size, audio_size)

        self.visual_linear = nn.Linear(visual_size, out_size)
        self.audio_linear = nn.Linear(audio_size, out_size)

        self.TEMPERATURE = 0.1

    @property
    def has_to_gather(self):
        return du.get_world_size() > 1 and self.use_global_batch

    def forward(self, visual, audio):
        batch_size = visual.size(0)

        out1 = F.normalize(self.visual_linear(visual), dim=-1)
        out2 = F.normalize(self.audio_linear(audio), dim=-1)

        gb_size = batch_size
        if self.has_to_gather:
            gb_size = batch_size * du.get_world_size()
            out1_large = du.diff_all_gather(out1)
            out2_large = du.diff_all_gather(out2)
            labels = (
                torch.arange(batch_size, device=out1.device)
                + du.get_rank() * batch_size
            )
        else:
            out1_large = out1
            out2_large = out2
            labels = torch.arange(batch_size, device=out1.device)

        out1_large = out1
        out2_large = out2
        labels = torch.arange(batch_size, device=out1.device)

        logits_ab = (
            out1.matmul(out2_large.transpose(0, 1))
        ) / self.TEMPERATURE

        logits_ba = (
            out2.matmul(out1_large.transpose(0, 1))
        ) / self.TEMPERATURE

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

    def infer(self, visual, audio):
        out1 = F.normalize(self.visual_linear(visual), dim=-1)
        out2 = F.normalize(self.audio_linear(audio), dim=-1)

        logits = torch.einsum('bc,bc->b', out1, out2)  # with aligned feature
        return logits

    def average_gradient(self):
        size = float(du.get_world_size())
        for param in self.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [
        top_max_k_correct[:k, :].view(-1).float().sum() for k in ks
    ]
    return topks_correct
