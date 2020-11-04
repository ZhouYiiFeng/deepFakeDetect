#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
---------------------------------
@ Author : JoeyF.Zhou           -
@ Home Page : www.zhoef.com     -
@ From : UESTC                  - 
---------------------------------
@ Date : 11/3/2020 7:04 PM
@ Project Name : FaceShifter-pytorch-master
@ Description :
    This code is writen by joey.
    @ function:
        ADDGenerator is composed by the ADDResBlk.
        The number of ADDResBlk is depend on the Attrs num.
        The first ADD input channel number is 64, output channel is 3. -> fake RGB image.
        The last ADD input channel number is 1024, output channel is 512.
            64, 128, 256, 512, 1024.
        The overflow ADD is designed as in_ch: 1024, ou_ch: 1024.
        Input:
            1. attrs_num from 64 to more.
            2. zid the identity code extract from facial module.
"""
import torch
import torch.nn as nn
import numpy as np
from my_ADD import ADDResBLK


class ADDGenerator(nn.Module):
    def __init__(self, attrs_num):
        super(ADDGenerator, self).__init__()
        self.attrs_num = attrs_num
        self.first_blk_inch = 64
        self.first_blk_och = 3
        self.final_blk_inch = 1024
        self.final_blk_och = 1024
        self.ADDBlks = nn.ModuleList()
        self.BUps = nn.ModuleList()
        self.first_transpose_conv = nn.ConvTranspose2d(in_channels=256,
                                                       out_channels=self.final_blk_inch,
                                                       kernel_size=2, stride=1, padding=0)

    def initADDGenerator(self):
        in_ch = self.first_blk_inch
        o_ch = self.first_blk_och
        attr_num = 0
        for attr_id in range(self.attrs_num):
            add = ADDResBLK(in_ch=in_ch, o_ch=o_ch)
            self.ADDBlks.append(add)
            if attr_id != 0:
                self.BUps.append(nn.UpsamplingBilinear2d())
            attr_num += 1
            if in_ch == 1024:
                break
            o_ch = in_ch
            in_ch *= 2

        while attr_num != self.attrs_num:
            add = ADDResBLK(in_ch=self.final_blk_inch, o_ch=self.final_blk_och)
            self.ADDBlks.append(add)
            self.BUps.append(nn.UpsamplingBilinear2d())
            attr_num += 1



    def forward(self, attrs, zid):

        for id, add in enumerate(self.ADDBlks):
            x = add(x[id])


