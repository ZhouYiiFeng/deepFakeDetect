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

    We design less conv transpose in this code in order to avoid the checkerboard phenomenon.
    We also found more 512 dim id code will lead to the collapse of our synthesis network, which shown in my_aei512.
    In this model:
        1. code = 512.
        2. the final size of Unet is set up to 4x4xC.
"""
import sys
sys.path.append('..')
sys.path.append('.')
from network.aei_v2_0.my_MultiLevelAttributesEncoder2 import MultiLevelAttributesEncoder
from network.aei_v2_0.my_ADDGenerator2 import ADDGenerator
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


class AETNet(nn.Module):
    def __init__(self, in_ch=3, in_H=256, in_W=256, zid_ch=512):
        super(AETNet, self).__init__()
        inner_zid_ch = 512
        self.multi_level_attribute_encoder = MultiLevelAttributesEncoder(in_ch=in_ch, in_H=in_H, in_W=in_W)
        self.fc = nn.Linear(zid_ch, inner_zid_ch)
        self.activate = nn.ReLU()
        self.add_generator = ADDGenerator(in_H, zid_ch)
        self.apply(init_weights)

    def forward(self, xt, iden_code):
        attrs = self.multi_level_attribute_encoder(xt)
        iden_code = self.activate(self.fc(iden_code))
        yt = self.add_generator(attrs, iden_code)
        return yt, attrs

    def getAttributes(self, image):
        return self.multi_level_attribute_encoder(image)