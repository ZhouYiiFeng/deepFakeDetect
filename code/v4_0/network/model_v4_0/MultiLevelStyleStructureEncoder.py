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
        # sface = F.interpolate(sface, scale_factor=0.5, mode='bilinear', align_corners=True)
        s_ldm = ldms[:, 0, :, :, :]
        t_ldm = ldms[:, 1, :, :, :]
        s_ldm = F.interpolate(s_ldm, scale_factor=0.5, mode='bilinear', align_corners=True)
        t_ldm = F.interpolate(t_ldm, scale_factor=0.5, mode='bilinear', align_corners=True)
        h_face_1, s_ldm, t_ldm = self.ssa_1(sface, sface_attrs[0], tface_pyr[0], s_ldm, t_ldm)  # in 256 out 128
        h_face_2, s_ldm, t_ldm = self.ssa_2(h_face_1, sface_attrs[1], tface_pyr[1], s_ldm, t_ldm)  # in 128 out 64
        mask1 = self.mask1(h_face_2)
        h_face_3, s_ldm, t_ldm = self.ssa_3(h_face_2, sface_attrs[2], tface_pyr[2], s_ldm, t_ldm)  # in 64 out 32
        mask2 = self.mask2(h_face_3)
        h_face_4, s_ldm, t_ldm = self.ssa_4(h_face_3, sface_attrs[3], tface_pyr[3], s_ldm, t_ldm)  # in 32 out 16
        mask3 = self.mask3(h_face_4)
        h_face_5, s_ldm, t_ldm = self.ssa_5(h_face_4, sface_attrs[4], tface_pyr[4], s_ldm, t_ldm)  # in 16 out 8
        mask4 = self.mask4(h_face_5)
        h_face_6, s_ldm, t_ldm = self.ssa_6(h_face_5, sface_attrs[5], tface_pyr[5], s_ldm, t_ldm)  # in 8 out 4
        mask5 = self.mask5(h_face_6)

        return [h_face_1, h_face_2, h_face_3, h_face_4, h_face_5, h_face_6], \
               [mask1, mask2, mask3, mask4, mask5]