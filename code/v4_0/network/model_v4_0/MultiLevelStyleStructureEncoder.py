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
from SSAResBLK import SSAResBLK


class MultiLevelStyleStructureEncoder(nn.Module):
    def __init__(self, in_ch):
        super(MultiLevelStyleStructureEncoder, self).__init__()
        self.ssa_1 = SSAResBLK(in_ch, 32, 32)
        self.ssa_2 = SSAResBLK(32, 64, 64)
        self.mask1 = nn.Conv2d(64, 1, kernel_size=1, stride=1)

        self.ssa_3 = SSAResBLK(64, 128, 128)
        self.mask2 = nn.Conv2d(128, 1, kernel_size=1, stride=1)

        self.ssa_4 = SSAResBLK(128, 256, 256)
        self.mask3 = nn.Conv2d(256, 1, kernel_size=1, stride=1)

        self.ssa_5 = SSAResBLK(256, 512, 512)
        self.mask4 = nn.Conv2d(512, 1, kernel_size=1, stride=1)

        self.ssa_6 = SSAResBLK(512, 512, 512)
        self.mask5 = nn.Conv2d(512, 1, kernel_size=1, stride=1)

    def forward(self, sface, sface_attrs, tface_pyr, ldms):
        sface = F.interpolate(sface, scale_factor=0.5, mode='bilinear', align_corners=True)
        h_face_1 = self.ssa_1(sface, sface_attrs[0], tface_pyr[0], ldms)
        h_face_2 = self.ssa_2(h_face_1, sface_attrs[1], tface_pyr[1], ldms)
        mask1 = self.mask1(h_face_2)
        h_face_3 = self.ssa_3(h_face_2, sface_attrs[2], tface_pyr[2], ldms)
        mask2 = self.mask1(h_face_3)
        h_face_4 = self.ssa_4(h_face_3, sface_attrs[3], tface_pyr[3], ldms)
        mask3 = self.mask1(h_face_4)
        h_face_5 = self.ssa_5(h_face_4, sface_attrs[4], tface_pyr[4], ldms)
        mask4 = self.mask1(h_face_5)
        h_face_6 = self.ssa_6(h_face_5, sface_attrs[5], tface_pyr[5], ldms)
        mask5 = self.mask1(h_face_6)

        return [h_face_1, h_face_2, h_face_3, h_face_4, h_face_5, h_face_6], \
               [mask1, mask2, mask3, mask4, mask5]