#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
---------------------------------
@ Author : JoeyF.Zhou           -
@ Home Page : www.zhoef.com     -
@ From : UESTC                  - 
---------------------------------
@ Date : 11/3/2020 12:57 PM
@ Project Name : FaceShifter-pytorch-master
@ Description :
    This code is writen by joey.
    @ function:
        Input:
            1. Given the attribution provider image Xt (UNet-like)
            2. Given the identity code extracted from the Xs (facial module)
        Output:
            1. Attrs list of attribute image Xt (N-layers)
            2. The synthesis Ys,t fake image.
"""
import sys
sys.path.append('/mnt/hdd2/std/zyf/dfke/faceGen/FaceShifter-pytorch/network')
from my_MultiLevelAttributesEncoder2 import MultiLevelAttributesEncoder
from my_ADDGenerator2 import ADDGenerator
import torch.nn as nn



class AETNet(nn.Module):
    def __init__(self, in_ch=3, in_H=256, in_W=256, zid_ch=512):
        super(AETNet, self).__init__()
        inner_zid_ch = 256
        self.multi_level_attribute_encoder = MultiLevelAttributesEncoder(in_ch=in_ch, in_H=in_H, in_W=in_W)
        self.fc = nn.Linear(zid_ch, inner_zid_ch)
        self.activate = nn.ReLU()
        self.add_generator = ADDGenerator(in_H, inner_zid_ch)

    def forward(self, xt, iden_code):
        attrs = self.multi_level_attribute_encoder(xt)
        iden_code = self.activate(self.fc(iden_code))
        yt = self.add_generator(attrs, iden_code)
        return yt, attrs

    def getAttributes(self, image):
        return self.multi_level_attribute_encoder(image)