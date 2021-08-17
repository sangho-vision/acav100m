"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 2D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        ResNetBasicHead takes the single audio pathway as input.
        Args:
            dim_in (int): the channel dimension of the input to the
                ResNetHead.
            num_classes (int): the channel dimension of the output to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of an avg pooling,
                frequency pool kernel size, time pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        self.avg_pool = nn.AvgPool2d(pool_size, stride=1)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=-1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, input):
        x = self.avg_pool(input)
        # (N, C, F, T) -> (N, F, T, C).
        x = x.permute((0, 2, 3, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2])

        x = x.view(x.shape[0], -1)
        return x


class ResNetPoolingHead(nn.Module):
    """
    ResNe(X)t 2D Pooling head.
    This layer performs an average pooling.
    """

    def __init__(
        self,
        pool_size,
    ):
        """
        ResNetPoolingHead takes the single audio pathway as input.

        Args:
            pool_size (list): the list of kernel sizes of an avg pooling,
                frequency pool kernel size, time pool kernel size in order.
        """
        super(ResNetPoolingHead, self).__init__()
        self.avg_pool = nn.AvgPool2d(pool_size, stride=1)

    def forward(self, input):
        x = self.avg_pool(input)
        x = x.view(x.shape[0], -1)
        return x
