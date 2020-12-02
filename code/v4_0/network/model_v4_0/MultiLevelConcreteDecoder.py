#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
---------------------------------
@ Author : JoeyF.Zhou           -
@ Home Page : www.zhoef.com     -
@ From : UESTC                  - 
---------------------------------
@ Date : 2020/11/25 21:47
@ Project Name : FaceShifter-pytorch-master
@ Description :
    This code is writen by joey.
    @ function:
    
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class UpSampleConv(nn.Module):
    def __init__(self, in_ch, ou_ch, kernel_size=4, stride=2, padding=1):
        super(UpSampleConv, self).__init__()
        self.t_conv = nn.ConvTranspose2d(in_ch, ou_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(ou_ch)
        self.activate = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, skip=None):
        x = self.t_conv(input)
        x = self.bn(x)
        x = self.activate(x)
        if skip is not None:
            return torch.cat((x, skip), dim=1)
        return x


class MultiLevelConcreteDecoder(nn.Module):
    def __init__(self):
        super(MultiLevelConcreteDecoder, self).__init__()
        self.nfc = 32  # the second channel num.
        self.conv_t1 = UpSampleConv(512, 512)
        self.conv_t2 = UpSampleConv(1024, 256)
        self.conv_t3 = UpSampleConv(512, 128)
        self.conv_t4 = UpSampleConv(256, 64)
        self.conv_t5 = UpSampleConv(128, 32)
        self.conv_t6 = UpSampleConv(64, 32)
        self.conv = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, bg_attrs, mix_face_attrs):
        bg_attrs = bg_attrs[::-1]
        mix_face_attrs = mix_face_attrs[::-1]
        x = self.conv(torch.cat([mix_face_attrs[0], bg_attrs[0]], dim=1))  # 512 + 512 out 512 same size 128
        x = self.conv_t1(x, mix_face_attrs[1] + bg_attrs[1])  # 512 512 out 512 8 8
        x = self.conv_t2(x, mix_face_attrs[2] + bg_attrs[2])  #     out 256, 16, 16
        x = self.conv_t3(x, mix_face_attrs[3] + bg_attrs[3])  #     out 128, 32, 32
        x = self.conv_t4(x, mix_face_attrs[4] + bg_attrs[4])  #     out 64, 64, 64
        x = self.conv_t5(x, mix_face_attrs[5] + bg_attrs[5])  #     out 32, 128, 128

        # mix_final = F.interpolate(mix_face_attrs[5], scale_factor=2, mode='bilinear', align_corners=True)
        # bg_final = F.interpolate(bg_attrs[5], scale_factor=2, mode='bilinear', align_corners=True)
        # x = self.conv_t6(x, mix_final + bg_final)  #     out 32, 256, 256
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_t6(x)
        x = self.conv1(x)
        x = F.tanh(x)
        return x