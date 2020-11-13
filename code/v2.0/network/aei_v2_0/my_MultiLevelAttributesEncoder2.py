#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
---------------------------------
@ Author : JoeyF.Zhou           -
@ Home Page : www.zhoef.com     -
@ From : UESTC                  - 
---------------------------------
@ Date : 11/5/2020 7:01 PM
@ Project Name : FaceShifter-pytorch-master
@ Description :
    This code is writen by joey.
    @ function:
    Different from my_MultiLevelAttributesEncoder, this is designed by the shape.
        Input:
            1. The attribution provider Xt
            2. The number of attributes: N
        Output
            1. A list of N attributes according to the N.

            The Network is like:
            3, H, W
            32, H/2, W/2
            C_H/4, H/4, W/4     --->C_Transpose (C_H/4 * 2, C_h/4 / 2)
            ...                            | --->attr: C_H/4 * 2, H/4, W/4
            C_4, 4, 4
            C_4, 2, 2
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


class DownSampleConv(nn.Module):
    def __init__(self, in_ch, ou_ch, kernel_size=4, stride=2, padding=1):
        super(DownSampleConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, ou_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(ou_ch)
        self.activate = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.activate(x)
        return x


class UpSampleConv(nn.Module):
    def __init__(self, in_ch, ou_ch, kernel_size=4, stride=2, padding=1):
        super(UpSampleConv, self).__init__()
        self.t_conv = nn.ConvTranspose2d(in_ch, ou_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(ou_ch)
        self.activate = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input):
        x = self.t_conv(input)
        x = self.bn(x)
        x = self.activate(x)
        return x


class MultiLevelAttributesEncoder(nn.Module):
    def __init__(self, in_ch, in_H, in_W, N=6):
        super(MultiLevelAttributesEncoder, self).__init__()
        self.down_convs = nn.ModuleList()
        self.transpose_convs = nn.ModuleList()
        self.nfc = 32  # the second channel num.
        self.input_channel = in_ch
        self.num_attr_extract_layers = N
        self.final_channel = 1024
        self.layer_num = 0
        self.in_H = in_H
        self.in_W = in_W
        assert self.in_W == self.in_H
        assert self.in_H % 2 == 0 and self.in_W % 2 == 0
        assert self.final_channel == self.nfc * pow(2, N-1)
        self.initUNetLikeExtractor()

    def getLayerNum(self):
        return self.layer_num

    def initUNetLikeExtractor(self):
        o_ch = self.nfc
        in_H = self.in_H
        in_W = self.in_W
        self.down_convs.append(DownSampleConv(self.input_channel, o_ch)) # first
        in_H = in_H // 2
        while in_H != 8:
            self.down_convs.append(DownSampleConv(o_ch, o_ch*2))
            self.transpose_convs.append(UpSampleConv(o_ch*4, o_ch))
            o_ch *= 2
            in_H //= 2
            in_W //= 2
        self.down_convs.append(DownSampleConv(o_ch, o_ch*2))
        self.transpose_convs.append(UpSampleConv(o_ch*2, o_ch))
        self.transpose_convs = self.transpose_convs[::-1]
        # self.apply(init_weights)

    def forward(self, input):
        ds_features = []
        attrs = []
        x = input
        for layer_id in range(len(self.down_convs)):  # downSample times, the downSp is one more than upSp
            x = self.down_convs[layer_id](x)
            ds_features.append(x)
        ds_features = ds_features[::-1]

        attrs.append(ds_features[0])
        for layer_id in range(len(self.transpose_convs)):
            prev_attr = attrs[-1]
            up_att = self.transpose_convs[layer_id](prev_attr)
            attr = torch.cat([up_att, ds_features[layer_id+1]], dim=1)
            attrs.append(attr)

        attrs.append(F.interpolate(attrs[-1], size=(self.in_W, self.in_H), mode='bilinear'))
        return attrs


if __name__ == '__main__':
    sqh = 256
    model = MultiLevelAttributesEncoder(in_ch=3, in_H=sqh, in_W=sqh)
    input = torch.rand(1, 3, sqh, sqh)
    output = model(input)
    print(model)
