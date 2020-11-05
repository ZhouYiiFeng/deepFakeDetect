#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
---------------------------------
@ Author : JoeyF.Zhou           -
@ Home Page : www.zhoef.com     -
@ From : UESTC                  - 
---------------------------------
@ Date : 11/3/2020 10:09 PM
@ Project Name : FaceShifter-pytorch-master
@ Description :
    This code is writen by joey.
    @ function:
    
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ADD(nn.Module):
    def __init__(self, in_ch, o_ch, attr_ch, zid_ch=256, mode='SPAD'):
        super(ADD, self).__init__()
        self.BN = nn.BatchNorm2d(in_ch)
        self.h_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, stride=1,kernel_size=3, padding=1)
        self.r_att_conv = nn.Conv2d(in_channels=attr_ch, out_channels=in_ch, stride=1,kernel_size=3, padding=1)
        self.beta_att_conv = nn.Conv2d(in_channels=attr_ch, out_channels=in_ch, stride=1,kernel_size=3, padding=1)
        self.zid_FC1 = nn.Linear(zid_ch, in_ch)
        self.zid_FC2 = nn.Linear(zid_ch, in_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, attr, zid, h):
        h_bn = self.BN(h)
        mask = self.sigmoid(self.h_conv(h_bn))
        r_att = self.r_att_conv(attr)
        beta_att = self.beta_att_conv(attr)
        Ak = h_bn * r_att + beta_att

        B, C, H, W = h.size()
        r_id = self.zid_FC1(zid)
        beta_id = self.zid_FC2(zid)
        r_id = r_id.reshape(B, -1, 1, 1).expand_as(h)
        beta_id = beta_id.reshape(B, -1, 1, 1).expand_as(h)
        Ik = h_bn * r_id + beta_id

        hout = (1-mask) * Ak + mask * Ik
        return hout


class ADDResBLK(nn.Module):
    def __init__(self, in_ch, o_ch, attr_ch, zid_ch=256):
        super(ADDResBLK, self).__init__()
        self.mode = True
        self.add1 = ADD(in_ch, o_ch, attr_ch, zid_ch)
        self.add2 = ADD(in_ch, o_ch, attr_ch, zid_ch)
        self.conv31 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, stride=1, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(in_channels=in_ch, out_channels=o_ch, stride=1, kernel_size=3, padding=1)
        self.activate = nn.ReLU()
        if in_ch != o_ch:
            self.add3 = ADD(in_ch, o_ch, attr_ch, zid_ch)
            self.mode = False
        if not self.mode:
            self.conv33 = nn.Conv2d(in_channels=in_ch, out_channels=o_ch, stride=1, kernel_size=3, padding=1)

    def forward(self, attr, zid, hprev):
        mid_h = self.add1(attr, zid, hprev)
        mid_h = self.activate(self.conv31(mid_h))
        mid_h = self.add2(attr, zid, mid_h)
        mid_h = self.activate(self.conv32(mid_h))
        if not self.mode:
            app_h = self.add3(attr, zid, hprev)
            app_h = self.activate(self.conv33(app_h))
            return mid_h + app_h
        else:
            return mid_h + hprev
