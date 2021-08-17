"""Audio models."""

import torch.nn as nn


def get_trans_func(name):
    """
    Retrieves the transformation module by name.
    """
    trans_funcs = {
        "bottleneck_transform": BottleneckTransform,
        "basic_transform": BasicTransform,
    }
    assert (
        name in trans_funcs.keys()
    ), "Transformation function '{}' not supported".format(name)
    return trans_funcs[name]


class BasicTransform(nn.Module):
    """
    Basic transformation: 3x3 or [3x1, 1x3], 3x3.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner=None,
        num_groups=1,
        stride_1x1=None,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        separable=False,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            dim_out (int): the channel dimension of the output.
            stride (int): the stride of the ResBlock.
            dim_inner (None): the inner dimension would not be used in
                BasicTransform.
            num_groups (int): number of groups for the convolution. Number of
                group is always 1 for BasicTransform.
            stride_1x1 (None): stride_1x1 will not be used in BasicTransform.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for normalization.
            bn_mmt (float): momentum for batch norm.
            dilation (int): size of dilation. Not used in BasicTransform.
            separable (bool): if True, divide 3x3 kernel into 3x1, 1x3
        """
        super(BasicTransform, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self.separable = separable
        self._construct(dim_in, dim_out, stride)

    def _construct(self, dim_in, dim_out, stride):
        if self.separable:
            # 3x1, BN, ReLU.
            self.a1 = nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=[3, 1],
                stride=[stride, 1],
                padding=[1, 0],
                bias=False,
            )
            self.a1_bn = nn.BatchNorm2d(
                dim_out, eps=self._eps, momentum=self._bn_mmt
            )
            self.a1_relu = nn.ReLU(inplace=self._inplace_relu)

            # 1x3, BN, ReLU.
            self.a2 = nn.Conv2d(
                dim_out,
                dim_out,
                kernel_size=[1, 3],
                stride=[1, stride],
                padding=[0, 1],
                bias=False,
            )
            self.a2_bn = nn.BatchNorm2d(
                dim_out, eps=self._eps, momentum=self._bn_mmt
            )
            self.a2_relu = nn.ReLU(inplace=self._inplace_relu)
        else:
            # 3x3, BN, ReLU.
            self.a = nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=[3, 3],
                stride=[stride, stride],
                padding=[1, 1],
                bias=False,
            )
            self.a_bn = nn.BatchNorm2d(
                dim_out, eps=self._eps, momentum=self._bn_mmt
            )
            self.a_relu = nn.ReLU(inplace=self._inplace_relu)

        # 3x3, BN.
        self.b = nn.Conv2d(
            dim_out,
            dim_out,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            bias=False,
        )
        self.b_bn = nn.BatchNorm2d(
            dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.b_bn.transform_final_bn = True

    def forward(self, x):
        if self.separable:
            x = self.a1(x)
            x = self.a1_bn(x)
            x = self.a1_relu(x)
            x = self.a2(x)
            x = self.a2_bn(x)
            x = self.a2_relu(x)
        else:
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

        x = self.b(x)
        x = self.b_bn(x)
        return x


class BottleneckTransform(nn.Module):
    """
    Bottleneck transformation: 1x1, 3x3 or [3x1, 1x3], 1x1.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        separable=False,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            dim_out (int): the channel dimension of the output.
            stride (int): the stride of the ResBlock.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for normalization.
            bn_mmt (float): momentum for batch norm.
            dilation (int): size of dilation.
            separable (bool): if True, divide 3x3 kernel into 3x1, 1x3
        """
        super(BottleneckTransform, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1x1
        self.separable = separable
        self._construct(
            dim_in, dim_out, stride, dim_inner, num_groups, dilation
        )

    def _construct(
        self, dim_in, dim_out, stride, dim_inner, num_groups, dilation
    ):
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        # 1x1, BN, ReLU.
        self.a = nn.Conv2d(
            dim_in,
            dim_inner,
            kernel_size=[1, 1],
            stride=[1, str1x1],
            padding=[0, 0],
            bias=False,
        )
        self.a_bn = nn.BatchNorm2d(
            dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)

        if self.separable:
            # 3x1, BN, ReL.U
            self.b1 = nn.Conv2d(
                dim_inner,
                dim_inner,
                [3, 1],
                stride=[str3x3, 1],
                padding=[dilation, 0],
                groups=num_groups,
                bias=False,
                dilation=[dilation, 1],
            )
            self.b1_bn = nn.BatchNorm2d(
                dim_inner, eps=self._eps, momentum=self._bn_mmt
            )
            self.b1_relu = nn.ReLU(inplace=self._inplace_relu)

            # 1x3, BN, ReLU.
            self.b2 = nn.Conv2d(
                dim_inner,
                dim_inner,
                [1, 3],
                stride=[1, str3x3],
                padding=[0, dilation],
                groups=num_groups,
                bias=False,
                dilation=[1, dilation],
            )
            self.b2_bn = nn.BatchNorm2d(
                dim_inner, eps=self._eps, momentum=self._bn_mmt
            )
            self.b2_relu = nn.ReLU(inplace=self._inplace_relu)
        else:
            # 3x3, BN, ReLU.
            self.b = nn.Conv2d(
                dim_inner,
                dim_inner,
                [3, 3],
                stride=[str3x3, str3x3],
                padding=[dilation, dilation],
                groups=num_groups,
                bias=False,
                dilation=[dilation, dilation],
            )
            self.b_bn = nn.BatchNorm2d(
                dim_inner, eps=self._eps, momentum=self._bn_mmt
            )
            self.b_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x1, BN.
        self.c = nn.Conv2d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            bias=False,
        )
        self.c_bn = nn.BatchNorm2d(
            dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.c_bn.transform_final_bn = True

    def forward(self, x):
        # Explicitly forward every layer.
        # Branch2a.
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)

        # Branch2b.
        if self.separable:
            x = self.b1(x)
            x = self.b1_bn(x)
            x = self.b1_relu(x)
            x = self.b2(x)
            x = self.b2_bn(x)
            x = self.b2_relu(x)
        else:
            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)

        # Branch2c
        x = self.c(x)
        x = self.c_bn(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        trans_func,
        dim_inner,
        num_groups=1,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        separable=False,
    ):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimension of the input.
            dim_out (int): the channel dimension of the output.
            stride (int): the stride of the ResBlock.
            trans_func (string): transform function to be used to construct each
                ResBlock.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for normalization.
            bn_mmt (float): momentum for batch norm.
            dilation (int): size of dilation.
            separable (bool): if True, divide 3x3 kernel into 3x1, 1x3
        """
        super(ResBlock, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._construct(
            dim_in,
            dim_out,
            stride,
            trans_func,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
            separable
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        trans_func,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
        separable
    ):
        # Use skip connection with projection if dim or resolution changes.
        if (dim_in != dim_out) or (stride != 1):
            self.branch1 = nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=[stride, stride],
                padding=0,
                bias=False,
                dilation=1,
            )
            self.branch1_bn = nn.BatchNorm2d(
                dim_out, eps=self._eps, momentum=self._bn_mmt
            )
        self.branch2 = trans_func(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=dilation,
            separable=separable,
            eps=self._eps,
            bn_mmt=self._bn_mmt,
        )
        self.relu = nn.ReLU(self._inplace_relu)

    def forward(self, x):
        if hasattr(self, "branch1"):
            x = self.branch1_bn(self.branch1(x)) + self.branch2(x)
        else:
            x = x + self.branch2(x)
        x = self.relu(x)
        return x


class ResStage(nn.Module):
    """
    Stage of 2D ResNet.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        num_blocks,
        dim_inner,
        num_groups,
        dilation,
        trans_func_name="bottleneck_transform",
        stride_1x1=False,
        inplace_relu=True,
        separable=False,
        eps=1e-5,
        bn_mmt=0.1,
    ):
        """
        ResStage builds the single audio stream.
        Args:
            dim_in (int): the channel dimension of the input.
            dim_out (int): the channel dimension of the output.
            stride (int): the stride of the ResBlock.
            num_blocks (int): the number of blocks.
            dim_inner (int): the inner channel dimension of the input.
            num_groups (int): the number of groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            dilation (int): the size of dilation.
            trans_func_name (string): name of the the transformation function applied
                on the network.
            separable (bool): if True, divide 3x3 kernel into 3x1, 1x3
            eps (float): epsilon for normalization.
            bn_mmt (float): momentum for batch norm.
        """
        super(ResStage, self).__init__()
        self.num_blocks = num_blocks
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            trans_func_name,
            stride_1x1,
            inplace_relu,
            dilation,
            separable,
            eps,
            bn_mmt,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        trans_func_name,
        stride_1x1,
        inplace_relu,
        dilation,
        separable,
        eps,
        bn_mmt,
    ):
        for i in range(self.num_blocks):
            # Retrieve the transformation function.
            trans_func = get_trans_func(trans_func_name)
            # Construct the block.
            res_block = ResBlock(
                dim_in if i == 0 else dim_out,
                dim_out,
                stride if i == 0 else 1,
                trans_func,
                dim_inner,
                num_groups,
                stride_1x1=stride_1x1,
                inplace_relu=inplace_relu,
                dilation=dilation,
                separable=separable,
                eps=eps,
                bn_mmt=bn_mmt,
            )
            self.add_module("res{}".format(i), res_block)

    def forward(self, x):
        for i in range(self.num_blocks):
            m = getattr(self, "res{}".format(i))
            x = m(x)

        return x
