import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.nn import init

import math
from PIL import Image
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.misc

from lib.gcn_lib import Grapher as GCB


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, groups=1):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


class UCB(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, groups=1, activation='relu'):
        super(UCB, self).__init__()

        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu6':
            self.activation = nn.ReLU6(inplace=True)
        elif activation == 'hardswish':
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        # self.up = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=True),
        #     nn.BatchNorm2d(ch_in),
        #     self.activation,
        #     nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
        # )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ScConv(ch_in),
            nn.BatchNorm2d(ch_in),
            self.activation,
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class trans_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=4, stride=2, padding=1, groups=32):
        super(trans_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                               bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        activation = 'relu'
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            # nn.GroupNorm(1,1),
            nn.Sigmoid()
        )

        if (activation == 'leakyrelu'):
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif (activation == 'gelu'):
            self.activation = nn.GELU()
        elif (activation == 'relu6'):
            self.activation = nn.ReLU6(inplace=True)
        elif (activation == 'hardswish'):
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.ratio = ratio
        activation = 'relu'
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // self.ratio, 1, bias=False)

        if (activation == 'leakyrelu'):
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif (activation == 'gelu'):
            self.activation = nn.GELU()
        elif (activation == 'relu6'):
            self.activation = nn.ReLU6(inplace=True)
        elif (activation == 'hardswish'):
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        self.fc2 = nn.Conv2d(in_planes // self.ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))
        # print(x.shape)
        max_pool_out = self.max_pool(x)

        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out)


class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out


class deformable_LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5, 5), padding=2, groups=dim)
        self.conv_spatial = DeformConv(dim, kernel_size=(7, 7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class deformable_Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = deformable_LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
        self.conv = nn.Conv2d(d_model, 1, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        x = self.conv(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        # print("seb的x的输入", x.shape)
        batch_size, channels, _, _ = x.size()
        y = self.squeeze(x).view(batch_size, channels)
        # print("y的输入", y.shape)
        y = self.excitation(y).view(batch_size, channels, 1, 1)
        out = x * y
        out = self.conv1(out)
        # print("seb的y的输出", x.shape)
        # print("seb的输出", out.shape)
        return out


class SPA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SPA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.SEBlock = SEBlock(768)

    def forward(self, x):
        # print("spa的输入", x.shape)
        # seb = self.SEBlock(x)
        # print("seb的输入", seb.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        # print("spa的输出", self.sigmoid(x).shape)
        return self.sigmoid(x)


class CUP(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64]):
        super(CUP, self).__init__()

        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])

        self.Up3 = up_conv(ch_in=channels[0], ch_out=channels[1])
        self.ConvBlock3 = conv_block(ch_in=2 * channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1], ch_out=channels[2])
        self.ConvBlock2 = conv_block(ch_in=2 * channels[2], ch_out=channels[2])

        self.Up1 = up_conv(ch_in=channels[2], ch_out=channels[3])
        self.ConvBlock1 = conv_block(ch_in=2 * channels[3], ch_out=channels[3])

    def forward(self, x, skips):
        d4 = self.ConvBlock4(x)

        # decoding + concat path
        d3 = self.Up3(d4)
        d3 = torch.cat((skips[0], d3), dim=1)

        d3 = self.ConvBlock3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((skips[1], d2), dim=1)
        d2 = self.ConvBlock2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((skips[2], d1), dim=1)
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1


class CASCADE_Cat(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64]):
        super(CASCADE_Cat, self).__init__()

        self.Conv_1x1 = nn.Conv2d(channels[0], channels[0], kernel_size=1, stride=1, padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])

        self.Up3 = up_conv(ch_in=channels[0], ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1], F_l=channels[1], F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=2 * channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1], ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2], F_l=channels[2], F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=2 * channels[2], ch_out=channels[2])

        self.Up1 = up_conv(ch_in=channels[2], ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3], F_l=channels[3], F_int=int(channels[3] / 2))
        self.ConvBlock1 = conv_block(ch_in=2 * channels[3], ch_out=channels[3])

        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(2 * channels[1])
        self.CA2 = ChannelAttention(2 * channels[2])
        self.CA1 = ChannelAttention(2 * channels[3])

        self.SA = SPA()

    def forward(self, x, skips):
        d4 = self.Conv_1x1(x)

        # CAM4
        d4 = self.CA4(d4) * d4
        d4 = self.SA(d4) * d4
        d4 = self.ConvBlock4(d4)

        # upconv3
        d3 = self.Up3(d4)

        # AG3
        x3 = self.AG3(g=d3, x=skips[0])

        # Concat 3
        d3 = torch.cat((x3, d3), dim=1)

        # CAM3
        d3 = self.CA3(d3) * d3
        d3 = self.SA(d3) * d3
        d3 = self.ConvBlock3(d3)

        # upconv2
        d2 = self.Up2(d3)

        # AG2
        x2 = self.AG2(g=d2, x=skips[1])

        # Concat 2
        d2 = torch.cat((x2, d2), dim=1)

        # CAM2
        d2 = self.CA2(d2) * d2
        d2 = self.SA(d2) * d2
        # print(d2.shape)
        d2 = self.ConvBlock2(d2)

        # upconv1
        d1 = self.Up1(d2)

        # print(skips[2])
        # AG1
        x1 = self.AG1(g=d1, x=skips[2])

        # Concat 1
        d1 = torch.cat((x1, d1), dim=1)

        # CAM1
        d1 = self.CA1(d1) * d1
        d1 = self.SA(d1) * d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1


def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out


def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))


def make_laplace(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        max_out = self.mlp(F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        channel_att_sum = avg_out + max_out

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


class EGA(nn.Module):
    def __init__(self, in_channels):
        super(EGA, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.cbam = CBAM(in_channels)

    def forward(self, edge_feature, x, pred):
        # print("edge_feature", edge_feature.shape)  # [2, 1, 128, 128]
        # print("x", x.shape)  # [2, 256, 16, 16]
        # print("pred", pred.shape)  # [2, 1, 16, 16]
        residual = x
        xsize = x.size()[2:]

        pred = torch.sigmoid(pred)

        # reverse attention
        background_att = 1 - pred
        background_x = x * background_att
        # print("background_x的维度", background_x.shape)

        # boudary attention
        edge_pred = make_laplace(pred, 1)
        # print("edge_pred的维度", edge_pred.shape)
        pred_feature = x * edge_pred
        # print("pred_feature的维度", pred_feature.shape)

        # high-frequency feature
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input

        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)
        # print("fusion_feature的维度", fusion_feature.shape)

        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map

        out = fusion_feature + residual
        # print("模块的输出", out.shape)
        out = self.cbam(out)
        return out


class PixLevelModule(nn.Module):
    def __init__(self, in_channels):
        super(PixLevelModule, self).__init__()
        self.middle_layer_size_ratio = 2
        self.conv_avg = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_avg = nn.ReLU(inplace=True)
        self.conv_max = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_max = nn.ReLU(inplace=True)
        self.bottleneck = nn.Sequential(
            nn.Linear(3, 3 * self.middle_layer_size_ratio),  # 2, 2*self.
            nn.ReLU(inplace=True),
            nn.Linear(3 * self.middle_layer_size_ratio, 1)
        )
        self.conv_sig = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    '''forward'''

    def forward(self, x):
        x_avg = self.conv_avg(x)
        x_avg = self.relu_avg(x_avg)
        x_avg = torch.mean(x_avg, dim=1)
        x_avg = x_avg.unsqueeze(dim=1)
        x_max = self.conv_max(x)
        x_max = self.relu_max(x_max)
        x_max = torch.max(x_max, dim=1).values
        x_max = x_max.unsqueeze(dim=1)
        x_out = x_max + x_avg
        x_output = torch.cat((x_avg, x_max, x_out), dim=1)
        x_output = x_output.transpose(1, 3)
        x_output = self.bottleneck(x_output)
        x_output = x_output.transpose(1, 3)
        y = x_output * x
        return y


class GCASCADE_Cat(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], drop_path_rate=0.0, img_size=224, k=11, padding=5, conv='mr',
                 gcb_act='gelu', activation='relu'):
        super(GCASCADE_Cat, self).__init__()

        #  Up-convolution block (UCB) parameters
        self.ucb_ks = 3
        self.ucb_pad = 1
        self.ucb_stride = 1
        self.activation = activation

        #  Graph convolution block (GCB) parameters
        self.padding = padding
        self.k = k  # neighbor num (default:9)
        self.conv = conv  # graph conv layer {edge, mr, sage, gin} # default mr
        self.gcb_act = gcb_act  # activation layer for graph convolution block {relu, prelu, leakyrelu, gelu, hswish}
        self.gcb_norm = 'batch'  # batch or instance normalization for graph convolution block {batch, instance}
        self.bias = True  # bias of conv layer True or False
        self.dropout = 0.0  # dropout rate
        self.use_dilation = True  # use dilated knn or not
        self.epsilon = 0.2  # stochastic epsilon for gcn
        self.use_stochastic = False  # stochastic for gcn, True or False
        self.drop_path = drop_path_rate
        self.reduce_ratios = [1, 1, 4, 2]
        self.dpr = [self.drop_path, self.drop_path, self.drop_path, self.drop_path]  # stochastic depth decay rule
        self.num_knn = [self.k, self.k, self.k, self.k]  # number of knn's k
        self.max_dilation = 18 // max(self.num_knn)
        self.HW = img_size // 4 * img_size // 4

        self.gcb4 = nn.Sequential(
            GCB(channels[0], self.num_knn[0], min(0 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[0], n=self.HW // (4 * 4 * 4),
                drop_path=self.dpr[0],
                relative_pos=True, padding=self.padding),
        )

        self.ucb3 = UCB(ch_in=channels[0], ch_out=channels[1], kernel_size=self.ucb_ks, stride=self.ucb_stride,
                        padding=self.ucb_pad, groups=channels[0], activation=self.activation)
        self.gcb3 = nn.Sequential(
            GCB(channels[1] * 2, self.num_knn[1], min(3 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[1], n=self.HW // (4 * 4),
                drop_path=self.dpr[1],
                relative_pos=True, padding=self.padding),
        )

        self.ucb2 = UCB(ch_in=channels[1] * 2, ch_out=channels[2], kernel_size=self.ucb_ks, stride=self.ucb_stride,
                        padding=self.ucb_pad, groups=channels[1], activation=self.activation)
        self.gcb2 = nn.Sequential(
            GCB(channels[2] * 2, self.num_knn[2], min(8 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[2], n=self.HW // (4),
                drop_path=self.dpr[2],
                relative_pos=True, padding=self.padding),
        )

        self.ucb1 = UCB(ch_in=channels[2] * 2, ch_out=channels[3], kernel_size=self.ucb_ks, stride=self.ucb_stride,
                        padding=self.ucb_pad, groups=channels[2], activation=self.activation)
        self.gcb1 = nn.Sequential(
            GCB(channels[3] * 2, self.num_knn[3], min(11 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[3], n=self.HW, drop_path=self.dpr[3],
                relative_pos=True, padding=self.padding),
        )

        self.spa = SPA()
        # self.dea4 = deformable_Attention(768)
        # self.dea3 = deformable_Attention(768)
        # self.dea2 = deformable_Attention(384)
        # self.dea1 = deformable_Attention(192)
        self.seb4 = SEBlock(768)
        self.seb3 = SEBlock(768)
        self.seb2 = SEBlock(384)
        self.seb1 = SEBlock(192)
        self.EGA1 = EGA(96)
        self.EGA2 = EGA(192)
        self.EGA3 = EGA(384)
        self.conv3 = nn.Conv2d(384, 1, 1)
        self.conv2 = nn.Conv2d(192, 1, 1)
        self.conv1 = nn.Conv2d(96, 1, 1)

    def forward(self, x, skips, edge_feature):
        # GCAM4
        d4 = self.gcb4(x)
        # print("d4", d4.shape)
        d44 = self.seb4(d4) * d4
        d4 = self.spa(d4) * d4
        d4 = d4 + d44
        # print("d4", d4.shape)

        # UCB3
        d3 = self.ucb3(d4)
        # print("d3", d3.shape)
        d33 = self.conv3(d3)
        # print("d3", d3.shape)  # [1, 320, 22, 22]
        # print("skips[0]", skips[0].shape)
        ega3 = self.EGA3(edge_feature, skips[0], d33)
        # print("ega3", ega3.shape)

        # Aggregation 3
        d3 = torch.cat((ega3, d3), dim=1)
        # print("d3", d3.shape)

        # GCAM3
        d3 = self.gcb3(d3)
        # print("d3", d3.shape)
        d33 = self.seb3(d3) * d3
        d3 = self.spa(d3) * d3
        d3 = d3 + d33
        # print("d3", d3.shape)

        # ucb2
        d2 = self.ucb2(d3)
        # print("d2", d2.shape)
        d22 = self.conv2(d2)
        ega2 = self.EGA2(edge_feature, skips[1], d22)
        # Aggregation 2
        d2 = torch.cat((ega2, d2), dim=1)
        # print("d2", d2.shape)

        # GCAM2
        d2 = self.gcb2(d2)
        # print("d2", d2.shape)
        d22 = self.seb2(d2) * d2
        # print("d22", d22.shape)
        d2 = self.spa(d2) * d2
        d2 = d2 + d22
        # print("d2", d2.shape)

        # ucb1
        d1 = self.ucb1(d2)
        # print("d1", d1.shape)
        d11 = self.conv1(d1)
        ega1 = self.EGA1(edge_feature, skips[2], d11)
        # Aggregation 1
        d1 = torch.cat((ega1, d1), dim=1)
        # print("d1", d1.shape)

        # GCAM1
        d1 = self.gcb1(d1)
        # print("d1", d1.shape)
        d11 = self.seb1(d1) * d1
        d1 = self.spa(d1) * d1
        d1 = d1 + d11
        # print("d1", d1.shape)

        return d4, d3, d2, d1

