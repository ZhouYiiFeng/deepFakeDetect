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
        self.attrs_num = attrs_num + 1 # one more for attr8 (bup)
        self.first_blk_inch = 64
        self.first_blk_och = 3

        self.last_attr_chn = 1024
        if attrs_num < 6:
            self.last_attr_chn = 1024 // 2**(6 - attrs_num)

        self.final_blk_inch = 1024
        self.final_blk_och = 1024
        self.ADDBlks = nn.ModuleList()
        self.BUp = nn.UpsamplingBilinear2d(scale_factor=2)
        self.first_transpose_conv = nn.ConvTranspose2d(in_channels=256,
                                                       out_channels=self.final_blk_inch,
                                                       kernel_size=2, stride=1, padding=0)
        self.initADDGenerator()

    def initADDGenerator(self):
        in_ch = self.first_blk_inch
        o_ch = self.first_blk_och
        attr_ch = self.first_blk_inch
        attr_num = 0
        for attr_id in range(self.attrs_num):
            add = ADDResBLK(in_ch=in_ch, o_ch=o_ch, attr_ch=attr_ch)
            self.ADDBlks.append(add)
            # if attr_id != 0:
            #     self.BUps.append(nn.UpsamplingBilinear2d())
            attr_num += 1
            if attr_id >= 1:  # z_att8 and z_att7 is 64 channel
                attr_ch *= 2
            o_ch = in_ch
            if in_ch == 1024:
                break
            in_ch *= 2

        while attr_num != self.attrs_num-1:
            if o_ch > self.final_blk_inch:  # final attr is 1024 channel num.
                add = ADDResBLK(in_ch=self.final_blk_inch, o_ch=self.final_blk_och,
                                attr_ch=attr_ch)
            else:
                add = ADDResBLK(in_ch=in_ch, o_ch=o_ch,
                                attr_ch=attr_ch)
                o_ch *= 2
                attr_ch *= 2
            self.ADDBlks.append(add)
            # self.BUps.append(nn.UpsamplingBilinear2d())
            attr_num += 1
        add = ADDResBLK(in_ch=self.final_blk_inch, o_ch=self.final_blk_och,
                        attr_ch=self.last_attr_chn)
        self.ADDBlks.append(add)
        self.ADDBlks = self.ADDBlks[::-1]

    def forward(self, attrs, zid):
        B,C = zid.shape
        h = self.first_transpose_conv(zid.reshape(B, -1, 1, 1))
        for id, add in enumerate(self.ADDBlks):
            h = add(attrs[id], zid, h)
            if id != self.attrs_num - 1:
                h = self.BUp(h)
        return h


if __name__ == '__main__':
    from my_MultiLevelAttributesEncoder import MultiLevelAttributesEncoder

    sqh = 64
    model = MultiLevelAttributesEncoder(in_ch=3, in_H=sqh, in_W=sqh)
    input = torch.rand(1, 3, sqh, sqh)
    zid = torch.rand(1, 256)
    attrs = model(input)
    model2 = ADDGenerator(model.layer_num)
    print(model2)
    yt = model2(attrs, zid)

