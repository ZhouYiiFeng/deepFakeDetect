#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
---------------------------------
@ Author : JoeyF.Zhou           -
@ Home Page : www.zhoef.com     -
@ From : UESTC                  - 
---------------------------------
@ Date : 2020/11/25 15:08
@ Project Name : FaceShifter-pytorch-master
@ Description :
    This code is writen by joey.
    @ function:
    
"""

import torch
import torch.nn as nn
import numpy as np


class DownSampleConv(nn.Module):
    def __init__(self, in_ch, ou_ch, kernel_size=4, stride=2, padding=1, normalize=False, dropout=0.0):
        super(DownSampleConv, self).__init__()
        self.layers = [nn.Conv2d(in_ch, ou_ch, kernel_size=kernel_size,
                                 stride=stride, padding=padding)]
        if normalize:
            self.layers.append(nn.InstanceNorm2d(ou_ch))
        if dropout:
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.downsample = nn.Sequential(*self.layers)

    def forward(self, input):
        x = self.downsample(input)
        return x


class MultiLevelAttributesEncoder(nn.Module):
    def __init__(self, in_ch):
        super(MultiLevelAttributesEncoder, self).__init__()
        self.down1 = DownSampleConv(in_ch, 32, normalize=False)  # 256->128
        self.down2 = DownSampleConv(32, 64)  # 128->64
        self.down3 = DownSampleConv(64, 128)  # 64->32
        self.down4 = DownSampleConv(128, 256)  # 32->16
        self.down5 = DownSampleConv(256, 512, dropout=0.5)  # 16->8
        self.down6 = DownSampleConv(512, 512, normalize=False, dropout=0.5)  # 8->4

    def forward(self, bcg, face_m=None):
        if face_m is not None:
            x = torch.cat([bcg, face_m], dim=1)
        else:
            x =bcg
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        return [d1, d2, d3, d4, d5, d6]
