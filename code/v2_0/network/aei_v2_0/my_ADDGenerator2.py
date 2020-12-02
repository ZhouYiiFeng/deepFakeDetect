#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
---------------------------------
@ Author : JoeyF.Zhou           -
@ Home Page : www.zhoef.com     -
@ From : UESTC                  - 
---------------------------------
@ Date : 11/5/2020 7:04 PM
@ Project Name : FaceShifter-pytorch-master
@ Description :
    This code is writen by joey.
    @ function:
    Different from my_ADDGenerator, this is designed by the shape.
        ADDGenerator is composed by the ADDResBlk.
        The number of ADDResBlk is depend on the Attrs num.
        The first ADD input channel number is 64, output channel is 3. -> fake RGB image.
        Last three ADD is same in out.
        Rest ADD is in/2 = out
        Input:
            1. input size of image tensor, HxW square required, therefore, one dim is enough.
            2. zid the identity code extract from facial module.

        In this code,
            we think the deep layer of our ADD-G depends more on the att image for its pose, hair, shape -> coarse style
            for middle styles such as facial, eyes depends both on the id and attr image.
            for Fine styles such as color scheme depends on the attr image.
        Different from baseline treat the id and attr equally in the ADDResBlk, we change the ratio and importance of id-
        and attr feature during the synthesis block.
"""
import torch
import torch.nn as nn
import numpy as np
from my_ADD import ADDResBLK


def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


class ADDGenerator(nn.Module):
    def __init__(self, in_size, zid_ch=512):
        super(ADDGenerator, self).__init__()
        # self.attrs_num = attrs_num + 1 # one more for attr8 (bup)
        self.first_blk_inch = 64
        self.first_blk_och = 3
        self.zid_ch = zid_ch
        self.in_size = in_size
        self.attr_num = np.log2(in_size)
        self.ADDBlks = nn.ModuleList()
        self.BUp = nn.UpsamplingBilinear2d(scale_factor=2)
        self.first_transpose_conv = nn.ConvTranspose2d(in_channels=self.zid_ch,
                                                       out_channels=self.in_size * 4,
                                                       kernel_size=2, stride=1, padding=0)
        self.initADDGenerator()

    def initADDGenerator(self):
        in_ch = self.in_size * 4
        o_ch = self.in_size * 4
        attr_ch = self.in_size * 4
        # first add fed with attr0
        size_attr = 2
        self.ADDBlks.append(ADDResBLK(in_ch=in_ch, o_ch=o_ch, attr_ch=attr_ch, zid_ch= self.zid_ch))
        attr_ch *= 2
        # second add fed with attr1
        self.ADDBlks.append(ADDResBLK(in_ch=in_ch, o_ch=o_ch, attr_ch=attr_ch, zid_ch= self.zid_ch))
        size_attr *= 2
        attr_ch //= 2

        while size_attr != self.in_size /2:
            self.ADDBlks.append(ADDResBLK(in_ch=in_ch, o_ch=o_ch, attr_ch=attr_ch, zid_ch= self.zid_ch))
            size_attr *= 2
            attr_ch //= 2
            in_ch = o_ch
            o_ch //= 2
        # last add -> output 3 channel
        add = ADDResBLK(in_ch=self.first_blk_inch, o_ch=self.first_blk_och,
                        attr_ch=self.first_blk_inch, zid_ch= self.zid_ch)
        self.ADDBlks.append(add)
        # self.apply(init_weights)

    def forward(self, attrs, zid):
        # B, C = zid.shape
        # h = self.first_transpose_conv(zid.reshape(B, -1, 1, 1))
        for id, add in enumerate(self.ADDBlks):
            h = add(attrs[id], zid, h)
            if id != len(self.ADDBlks) - 1:
                h = self.BUp(h)
        return torch.tanh(h)


if __name__ == '__main__':
    # from my_MultiLevelAttributesEncoder2 import MultiLevelAttributesEncoder

    sqh = 64
    # model = MultiLevelAttributesEncoder(in_ch=3, in_H=sqh, in_W=sqh)
    # input = torch.rand(1, 3, sqh, sqh)
    # zid = torch.rand(1, 256)
    # attrs = model(input)
    model2 = ADDGenerator(sqh)
    print(model2)
    # yt = model2(attrs, zid)

