#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
---------------------------------
@ Author : JoeyF.Zhou           -
@ Home Page : www.zhoef.com     -
@ From : UESTC                  - 
---------------------------------
@ Date : 2020/11/25 22:07
@ Project Name : FaceShifter-pytorch-master
@ Description :
    This code is writen by joey.
    @ function:
    
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SSA(nn.Module):
    def __init__(self, in_ch, face_attr_in, out_ch):
        super(SSA, self).__init__()
        self.ldmsConv1 = nn.Conv2d(68, face_attr_in, kernel_size=3, stride=1, padding=1)
        self.ldmsConv2 = nn.Conv2d(68, face_attr_in, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=face_attr_in + face_attr_in, out_channels=face_attr_in, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_ch, out_channels=face_attr_in, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=face_attr_in, out_channels=out_ch, kernel_size=3, stride=1, padding=1)

        self.spd_conv1 = nn.Conv2d(in_channels=3, out_channels=out_ch, kernel_size=1, stride=1, padding=0)
        self.spd_conv2 = nn.Conv2d(in_channels=3, out_channels=out_ch, kernel_size=1, stride=1, padding=0)

        self.norm = nn.InstanceNorm2d(out_ch, affine=False)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, pre_stage, sface, tface, s_ldm, t_ldm):
        # s_ldm = s_ldm, t_ldm[:, 0, :, :, :]
        # t_ldm = s_ldm, t_ldm[:, 1, :, :, :]
        x_s_l = self.ldmsConv1(s_ldm)
        x_t_l = self.ldmsConv2(t_ldm)
        sface = sface * x_s_l
        h = torch.cat([sface, x_t_l], dim=1)  # simplely add?
        h = self.relu(self.conv1(h))
        pre_stage = self.relu(self.conv2(pre_stage))
        h = pre_stage + h
        h = self.conv3(h)
        h_norm = self.norm(h)
        att_beta = self.spd_conv1(tface)
        att_gamma = self.spd_conv2(tface)
        A = att_gamma * h_norm + att_beta
        return A


class SSAResBLK(nn.Module):
    def __init__(self, in_ch, face_attr_in, out_ch):
        super(SSAResBLK, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ssa1 = SSA(in_ch, face_attr_in, out_ch)
        self.conv31 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.ssa2 = SSA(out_ch, face_attr_in, out_ch)
        self.conv32 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.h_downsp = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=4, stride=2, padding=1)
        self.conv_downsp = nn.Conv2d(in_channels=out_ch + 68, out_channels=68, kernel_size=4, stride=2, padding=1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        if in_ch!= out_ch:
            self.ssa3 = SSA(in_ch, face_attr_in, out_ch)
            self.conv33 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, pre_stage, sface_attr, tface, s_ldm, t_ldm):
        pre_stage = self.h_downsp(pre_stage)
        x = self.ssa1(pre_stage, sface_attr, tface, s_ldm, t_ldm)
        x = self.relu(x)
        x = self.conv31(x)
        x = self.ssa2(x, sface_attr, tface, s_ldm, t_ldm)
        x = self.relu(x)
        x = self.conv32(x)

        if self.in_ch != self.out_ch:
            pre_stage = self.ssa3(pre_stage, sface_attr, tface, s_ldm, t_ldm)
            pre_stage = self.relu(pre_stage)
            pre_stage = self.conv33(pre_stage)

        x = x + pre_stage
        x_tldm = torch.cat([x, t_ldm], dim=1)
        x_tldm = self.conv_downsp(x_tldm)

        x_sldm = torch.cat([x, s_ldm], dim=1)
        x_sldm = self.conv_downsp(x_sldm)

        return x, x_tldm, x_sldm

