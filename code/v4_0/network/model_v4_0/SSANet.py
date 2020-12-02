#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
---------------------------------
@ Author : JoeyF.Zhou           -
@ Home Page : www.zhoef.com     -
@ From : UESTC                  - 
---------------------------------
@ Date : 2020/11/25 15:06
@ Project Name : FaceShifter-pytorch-master
@ Description :
    This code is writen by joey.
    @ function:
    
"""
import sys
# import os
# print(os.path.abspath('.'))
# sys.path.append('/mnt/hdd2/std/zyf/dfke/faceGen/FaceShifter-pytorch/network')
sys.path.append('./network')
from MultiLevelAttributesEncoder import MultiLevelAttributesEncoder
from MultiLevelStyleStructureEncoder import MultiLevelStyleStructureEncoder
from MultiLevelConcreteDecoder import MultiLevelConcreteDecoder
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight.data)


class SSANet(nn.Module):
    def __init__(self, in_ch=3):
        super(SSANet, self).__init__()
        self.multi_level_attribute_encoder = MultiLevelAttributesEncoder(in_ch=in_ch + 1)
        self.multi_level_face_encoder = MultiLevelAttributesEncoder(in_ch=in_ch)
        self.multi_level_SSA_encoder = MultiLevelStyleStructureEncoder(in_ch=in_ch)
        self.multi_level_concrete_decoder = MultiLevelConcreteDecoder()
        self.apply(init_weights)

    def build_pyr(self, image, num=5):
        pyr = []
        pyr.append(F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=True))
        for i in range(num):
            t = F.interpolate(pyr[-1], scale_factor=0.5, mode='bilinear', align_corners=True)
            pyr.append(t)
        return pyr

    def forward(self, bcg, sface, tface, ldmm, mask):
        bg_attrs = self.multi_level_attribute_encoder(bcg, mask)
        sface_attrs = self.multi_level_face_encoder(sface)
        tface_pyr = self.build_pyr(tface, 6)
        mix_face_attrs, masks_pre_pyr = self.multi_level_SSA_encoder(sface, sface_attrs, tface_pyr, ldm)
        yt = self.multi_level_concrete_decoder(bg_attrs, mix_face_attrs)
        return yt


if __name__ == '__main__':
    # from my_MultiLevelAttributesEncoder2 import MultiLevelAttributesEncoder
    import torch
    # model = MultiLevelAttributesEncoder(in_ch=3, in_H=sqh, in_W=sqh)
    bcg = torch.rand(1, 3, 256, 256)
    mask = torch.rand(1, 1, 256, 256)
    sface = torch.rand(1, 3, 256, 256)
    tface = torch.rand(1, 3, 256, 256)
    ldm = torch.rand(1, 2, 68, 256, 256)
    # zid = torch.rand(1, 256)
    # attrs = model(input)
    model2 = SSANet()
    # print(model2)

    yt = model2(bcg, sface, tface, ldm, mask)