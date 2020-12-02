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

    def forward(self, input):
        x = self.t_conv(input)
        x = self.bn(x)
        x = self.activate(x)
        return x


class MultiLevelConcreteDecoder(nn.Module):
    def __init__(self):
        super(MultiLevelConcreteDecoder, self).__init__()
        self.nfc = 32  # the second channel num.
        self.conv_t1 = UpSampleConv(1024, 512)
        self.conv_t2 = UpSampleConv(512, 256)
        self.conv_t3 = UpSampleConv(256, 128)
        self.conv_t4 = UpSampleConv(126, 64)
        self.conv_t5 = UpSampleConv(64, 32)
        self.conv = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, bg_attrs, mix_face_attrs):
        x = self.conv(torch.cat([mix_face_attrs[0], bg_attrs[0]]))
        x = self.conv_t1(x, torch.cat([mix_face_attrs[1], bg_attrs[1]]))
        x = self.conv_t2(x, torch.cat([mix_face_attrs[2], bg_attrs[2]]))
        x = self.conv_t3(x, torch.cat([mix_face_attrs[3], bg_attrs[3]]))
        x = self.conv_t4(x, torch.cat([mix_face_attrs[4], bg_attrs[4]]))
        x = self.conv_t5(x, torch.cat([mix_face_attrs[5], bg_attrs[5]]))
        x = self.conv1(x)
        return x