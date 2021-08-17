import math

import torch
import torch.nn as nn
import torch.distributed as dist


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


def identity(x):
    return x


ACT2FN = {
    "identity": identity,
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
    "gelu_new": gelu_new,
    "mish": mish,
}


class FFNLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        intermediate_dim,
        output_dim,
        dropout=0.0,
        activation='relu',
        norm="batch_norm",
        eps=1e-5,
        bn_mmt=0.1,
    ):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(
            input_dim,
            intermediate_dim,
            bias=False if norm in ["layer_norm", "batch_norm"] else True,
        )
        if norm == "layer_norm":
            self.ln = nn.LayerNorm(intermediate_dim, eps=eps)
        elif norm == "batch_norm":
            self.bn = nn.BatchNorm1d(intermediate_dim, eps=eps, momentum=bn_mmt)
        self.dropout_func1 = nn.Dropout(dropout)
        self.dropout_func2 = nn.Dropout(dropout)
        self.activation = activation
        self.intermediate_act_fn = ACT2FN[activation]
        self.fc2 = nn.Linear(intermediate_dim, output_dim, bias=True)

    def forward(self, input, inplace=False):
        inter = self.dropout_func1(input)
        inter = self.fc1(inter)
        if hasattr(self, "ln"):
            inter = self.ln(inter)
        elif hasattr(self, "bn"):
            inter = self.bn(inter)
        if self.activation == "relu":
            inter_act = self.intermediate_act_fn(inter, inplace)
        else:
            inter_act = self.intermediate_act_fn(inter)
        out = self.dropout_func2(inter_act)
        return self.fc2(out)
