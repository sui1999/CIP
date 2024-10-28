import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

import logging

from lib.pvtv2 import *
from lib.decoders import *

from lib.maxxvit_4out import maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out


logger = logging.getLogger(__name__)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)



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


def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr



class MERIT_GCASCADE(nn.Module):
    def __init__(self, n_class=1, img_size_s1=(256, 256), img_size_s2=(224, 224), k=11, padding=5, conv='mr',
                 gcb_act='gelu', activation='relu', interpolation='bilinear', skip_aggregation='additive'):
        super(MERIT_GCASCADE, self).__init__()

        self.interpolation = interpolation
        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2
        self.skip_aggregation = skip_aggregation
        self.n_class = n_class

        # conv block to convert single channel to 3 channels
        self.conv_1cto3c = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # backbone network initialization with pretrained weight
        self.backbone1 = maxxvit_rmlp_small_rw_256_4out()  # [64, 128, 320, 512]
        # self.backbone2 = maxvit_rmlp_small_rw_224_4out()  # [64, 128, 320, 512]

        print('Loading:', './pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        state_dict1 = torch.load('./pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        self.backbone1.load_state_dict(state_dict1, strict=False)

        # print('Loading:', './pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
        # state_dict2 = torch.load('./pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
        # self.backbone2.load_state_dict(state_dict2, strict=False)

        print('Pretrain weights loaded.')

        self.channels = [768, 384, 192, 96]

        # decoder initialization 
        if self.skip_aggregation == 'additive':
            self.decoder1 = GCASCADE(channels=self.channels, img_size=img_size_s1[0], k=k, padding=padding, conv=conv,
                                     gcb_act=gcb_act, activation=activation)
            self.decoder2 = GCASCADE(channels=self.channels, img_size=img_size_s2[0], k=k, padding=padding, conv=conv,
                                     gcb_act=gcb_act, activation=activation)
        elif self.skip_aggregation == 'concatenation':
            self.decoder1 = GCASCADE_Cat(channels=self.channels, img_size=img_size_s1[0], k=k, padding=padding,
                                         conv=conv, gcb_act=gcb_act, activation=activation)
            # self.decoder2 = GCASCADE_Cat(channels=self.channels, img_size=img_size_s2[0], k=k, padding=padding,
            #                              conv=conv, gcb_act=gcb_act, activation=activation)
            self.channels = [self.channels[0], self.channels[1] * 2, self.channels[2] * 2, self.channels[3] * 2]
        else:
            print(
                'No implementation found for the skip_aggregation ' + self.skip_aggregation + '. Continuing with the default additive aggregation.')
            self.decoder1 = GCASCADE(channels=self.channels, img_size=img_size_s1[0], k=k, padding=padding, conv=conv,
                                     gcb_act=gcb_act, activation=activation)
            self.decoder2 = GCASCADE(channels=self.channels, img_size=img_size_s2[0], k=k, padding=padding, conv=conv,
                                     gcb_act=gcb_act, activation=activation)

        print('Model %s created, param count: %d' %
              ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder1.parameters()])))
        # print('Model %s created, param count: %d' %
        #       ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder2.parameters()])))

        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], n_class, 1)

        self.out_head4_in = nn.Conv2d(self.channels[3], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv_1cto3c(x)

        grayscale_img = rgb_to_grayscale(x)
        # print("灰度化", grayscale_img.shape)
        edge_feature = make_laplace_pyramid(grayscale_img, 5, 1)
        edge_feature = edge_feature[1]  # [1, 1, 176, 176] [1, 1, 224, 224] [1, 1, 128, 128]
        # print("edge_feature的维度", edge_feature.shape)

        # transformer backbone as encoder
        f1 = self.backbone1(F.interpolate(x, size=self.img_size_s1, mode=self.interpolation))
        # print([f1[3].shape,f1[2].shape,f1[1].shape,f1[0].shape])

        # decoder
        x11_o, x12_o, x13_o, x14_o = self.decoder1(f1[3], [f1[2], f1[1], f1[0]], edge_feature)

        # prediction heads  
        p11 = self.out_head1(x11_o)
        p12 = self.out_head2(x12_o)
        p13 = self.out_head3(x13_o)
        p14 = self.out_head4(x14_o)

        p14_in = self.out_head4_in(x14_o)
        p14_in = self.sigmoid(p14_in)

        p11 = F.interpolate(p11, scale_factor=32, mode=self.interpolation)
        p12 = F.interpolate(p12, scale_factor=16, mode=self.interpolation)
        p13 = F.interpolate(p13, scale_factor=8, mode=self.interpolation)
        p14 = F.interpolate(p14, scale_factor=4, mode=self.interpolation)

        # p14_in = F.interpolate(p14_in, scale_factor=4, mode=self.interpolation)
        # x_in = x * p14_in
        #
        # f2 = self.backbone2(F.interpolate(x_in, size=self.img_size_s2, mode=self.interpolation))
        #
        # skip1_0 = F.interpolate(f1[0], size=(f2[0].shape[-2:]), mode=self.interpolation)
        # skip1_1 = F.interpolate(f1[1], size=(f2[1].shape[-2:]), mode=self.interpolation)
        # skip1_2 = F.interpolate(f1[2], size=(f2[2].shape[-2:]), mode=self.interpolation)
        # skip1_3 = F.interpolate(f1[3], size=(f2[3].shape[-2:]), mode=self.interpolation)
        #
        # x21_o, x22_o, x23_o, x24_o = self.decoder2(f2[3] + skip1_3, [f2[2] + skip1_2, f2[1] + skip1_1, f2[0] + skip1_0], edge_feature)
        #
        # p21 = self.out_head1(x21_o)
        # p22 = self.out_head2(x22_o)
        # p23 = self.out_head3(x23_o)
        # p24 = self.out_head4(x24_o)
        #
        # # print([p21.shape,p22.shape,p23.shape,p24.shape])
        #
        # p21 = F.interpolate(p21, size=(p11.shape[-2:]), mode=self.interpolation)
        # p22 = F.interpolate(p22, size=(p12.shape[-2:]), mode=self.interpolation)
        # p23 = F.interpolate(p23, size=(p13.shape[-2:]), mode=self.interpolation)
        # p24 = F.interpolate(p24, size=(p14.shape[-2:]), mode=self.interpolation)

        p1 = p11
        p2 = p12
        p3 = p13
        p4 = p14
        p = p1 + p2 + p3 + p4
        # print([p1.shape,p2.shape,p3.shape,p4.shape])
        return p


if __name__ == '__main__':
    model = PVT_GCASCADE().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    p1, p2, p3, p4 = model(input_tensor)
    print(p1.size(), p2.size(), p3.size(), p4.size())
