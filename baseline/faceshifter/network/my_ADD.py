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


class ADDResBLK(nn.Module):
    def __init__(self, in_ch, o_ch):
        super(ADDResBLK, self).__init__()

    def forward(self, *input):
        pass