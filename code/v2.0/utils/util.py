#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
---------------------------------
@ Author : JoeyF.Zhou           -
@ Home Page : www.zhoef.com     -
@ From : UESTC                  - 
---------------------------------
@ Date : 11/6/2020 7:54 PM
@ Project Name : FaceShifter-pytorch-master
@ Description :
    This code is writen by joey.
    @ function:
    
"""
import torch
import torchvision

def hinge_loss(X, positive=True):
    if positive:
        return torch.relu(1-X).mean()
    else:
        return torch.relu(X+1).mean()


def get_grid_image(X):
    n = X.shape[0]
    X = X[:n+1]
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]) * 0.5 + 0.5
    return X


def make_image(Xs, Xt, Y):
    Xs = get_grid_image(Xs)
    Xt = get_grid_image(Xt)
    Y = get_grid_image(Y)
    return torch.cat((Xs, Xt, Y), dim=1)


def parse_gpu_id(str_gpuid):
    gpu_str = ""
    for id, gpu_id in enumerate(str_gpuid):
        if id == 0:
            gpu_str = str(gpu_id)
        else:
            gpu_str = gpu_id + ',' + str(gpu_id)
    return gpu_str
