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

import torch
import numpy as np
from my_MultiLevelAttributesEncoder import MultiLevelAttributesEncoder
from my_ADDGenerator import ADDGenerator
import torch.nn as nn


class AETNet(nn.Module):
    def __init__(self, in_ch=3, in_H=256, in_W=256):
        super(AETNet, self).__init__()
        self.multi_level_attribute_encoder = MultiLevelAttributesEncoder(in_ch=in_ch, in_H=in_H, in_W=in_W)
        self.add_generator = ADDGenerator(self.multi_level_attribute_encoder.layer_num)

    def forward(self, xt, iden_code):
        attrs = self.multi_level_attribute_encoder(xt)
        yt = self.add_generator(attrs, iden_code)
        return yt, attrs

    def getAttributes(self, image):
        return self.multi_level_attribute_encoder(image)