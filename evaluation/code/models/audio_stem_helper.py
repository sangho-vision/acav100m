"""ResNe(X)t 2D stem helper."""

import torch.nn as nn


class AudioModelStem(nn.Module):
    """
    Audio 2D stem module. Provides stem operations of Conv, BN, ReLU
    on input data tensor for the single audio pathway.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        separable=True,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel's size of the convolution in the stem
                layer. Frequency kernel size, time kernel size in order.
            stride (list): the stride size of the convolution in the stem
                layer. Frequency kernel stride, time kernel stirde in order.
            padding (list): the padding's size of the convolution in the stem
                layer. Frequency padding size, time padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for normalization.
            bn_mmt (float): momentum for batch norm.
            separable (bool): if True, divide kxk kernel into kx1, 1xk
        """
        super(AudioModelStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt

        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, separable)

    def _construct_stem(self, dim_in, dim_out, separable):
        stem = ResNetBasicStem(
            dim_in,
            dim_out,
            self.kernel,
            self.stride,
            self.padding,
            self.inplace_relu,
            self.eps,
            self.bn_mmt,
            separable,
        )
        self.add_module("stem", stem)

    def forward(self, x):
        x = self.stem(x)
        return x


class ResNetBasicStem(nn.Module):
    """
    ResNe(X)t 2D stem module.
    Performs Convolution, BN, and Relu.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        separable=True,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                Frequency kernel size, time kernel size in order.
            stride (list): the stride size of the convolution in the stem layer.
                Frequency kernel stride, time kernel stirde in order.
            padding (list): the padding size of the convolution in the stem
                layer. Frequency padding size, time padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for normalization.
            bn_mmt (float): momentum for batch norm.
            separable (bool): if True, divide kxk kernel into kx1, 1xk
        """
        super(ResNetBasicStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.separable = separable

        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        if self.separable:
            self.conv1 = nn.Conv2d(
                dim_in,
                dim_out,
                [self.kernel[0], 1],
                stride=[self.stride[0], 1],
                padding=[self.padding[0], 0],
                bias=False
            )
            self.bn1 = nn.BatchNorm2d(
                dim_out, eps=self.eps, momentum=self.bn_mmt
            )
            self.relu1 = nn.ReLU(self.inplace_relu)
            self.conv2 = nn.Conv2d(
                dim_out,
                dim_out,
                [1, self.kernel[1]],
                stride=[1, self.stride[1]],
                padding=[0, self.padding[1]],
                bias=False
            )
            self.bn2 = nn.BatchNorm2d(
                dim_out, eps=self.eps, momentum=self.bn_mmt
            )
            self.relu2 = nn.ReLU(self.inplace_relu)
        else:
            self.conv = nn.Conv2d(
                dim_in,
                dim_out,
                self.kernel,
                stride=self.stride,
                padding=self.padding,
                bias=False,
            )
            self.bn = nn.BatchNorm2d(
                dim_out, eps=self.eps, momentum=self.bn_mmt
            )
            self.relu = nn.ReLU(self.inplace_relu)

    def forward(self, x):
        if self.separable:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
        return x
