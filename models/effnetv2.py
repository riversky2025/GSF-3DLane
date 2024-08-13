import copy
import math
from functools import partial
from typing import Any, Callable, Optional, List, Sequence
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from models.submodule import *
from .mobilenetv3 import ConvNormActivation, SqueezeExcitation
from .mobilenetv3 import _make_divisible
import torch
import torch.fx
from torch import nn, Tensor
from typing import Any
from types import FunctionType

# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )




class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)





class EfficientNetFeature(nn.Module):
    def __init__(
        self,
            cfgs,
            width_mult,
            stereo,
            **kwargs: Any,
    ) -> None:
        """
        EfficientNet main class

        Args:
            inverted_residual_setting (List[MBConvConfig]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        self.cfgs = cfgs

        self.stereo = stereo
        layers: List[nn.Module] = []
        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers.append(conv_3x3_bn(3, input_channel, 2))
        # building inverted residual blocks
        block = MBConv
        output_channels=[]
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            stage: List[nn.Module] = []
            for i in range(n):
                stage.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
            output_channels.append(output_channel)
            layers.append(nn.Sequential(*stage))

        self.l0 = layers[0]
        self.l1 = layers[1]
        self.l2 = layers[2]
        self.l3 = layers[3]
        self.l4 = layers[4]
        self.l5 = layers[5]
        self.l6 = layers[6]
        # self.l7 = layers[7]

        self.feat_channel = 256
        fcg_channel=output_channels[2]+output_channels[4]+output_channels[5]
        if self.stereo:
            self.final_conv = nn.Sequential(convbn(288, 288, 5, 1, 2, 1),
                                            nn.ReLU(inplace=True),
                                            convbn(288, self.feat_channel, 3, 1, 1, 1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.feat_channel, self.feat_channel, kernel_size=1, padding=0,
                                                      stride=1, bias=False))
        else:
            self.final_conv = nn.Sequential(convbn(fcg_channel, 128, 5, 1, 2, 1),
                                            nn.ReLU(inplace=True),
                                            convbn(128, self.feat_channel, 3, 1, 1, 1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.feat_channel, self.feat_channel, kernel_size=1, padding=0,
                                                      stride=1, bias=False))
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:

        x = self.l0(x)      # 1/2
        l1 = self.l1(x)     # 1/2
        l2 = self.l2(l1)    # 1/4
        l3 = self.l3(l2)    # 1/8
        l4 = self.l4(l3)    # 1/16
        l5 = self.l5(l4)  # 1/2
        l6 = self.l6(l5)  # 1/4
        # l7 = self.l3(l6)  # 1/8

        if self.stereo:
            #############  1/2
            l2_interp = F.interpolate(l2, [l1.size()[2], l1.size()[3]], mode='bicubic', align_corners=True)
            l3_interp = F.interpolate(l3, [l1.size()[2], l1.size()[3]], mode='bicubic', align_corners=True)
            l4_interp = F.interpolate(l4, [l1.size()[2], l1.size()[3]], mode='bicubic', align_corners=True)
            feat = torch.cat((l1, l2_interp, l3_interp, l4_interp), dim=1)
        else:
            #############  1/4
            l3_interp = F.interpolate(l3, [l5.size()[2], l5.size()[3]], mode='bicubic', align_corners=True)
            l6_interp = F.interpolate(l6, [l5.size()[2], l5.size()[3]], mode='bicubic', align_corners=True)
            feat = torch.cat((l3_interp, l5, l6_interp), dim=1)

        x = self.final_conv(feat)

        return x



def efficientnet_feature(stereo, **kwargs: Any) -> EfficientNetFeature:
    # b6
    cfgs = [
        # t, c, n, s, SE
        [1, 8, 2, 1, 0],
        [4, 16, 3, 2, 0],
        [4, 64, 5, 2, 0],
        [4, 16, 7, 2, 1],
        [6, 32, 5, 1, 1],
        [6, 64, 3, 2, 1],

    ]
    width_mult = 1.8

    model = EfficientNetFeature(cfgs,width_mult,stereo,**kwargs)

    return model

