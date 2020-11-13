#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
---------------------------------
@ Author : JoeyF.Zhou           -
@ Home Page : www.zhoef.com     -
@ From : UESTC                  - 
---------------------------------
@ Date : 11/11/2020 5:06 PM
@ Project Name : FaceShifter-pytorch-master
@ Description :
    This code is writen by joey.
    @ function:
    
"""
from options.base_option import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # model
        self.parser.add_argument('--lr_G', type=float, default=1e-4, help='learning rate for generator')
        self.parser.add_argument('--lr_D', type=float, default=1e-4, help='learning rate for discriminator')
        self.parser.add_argument('--l_adv', type=float, default=1, help='the weight of l_adv')
        self.parser.add_argument('--l_att', type=float, default=10, help='the weight of l_att')
        self.parser.add_argument('--l_id', type=float, default=5, help='the weight of l_id ')
        self.parser.add_argument('--l_rec', type=float, default=10, help='the weight of l_rec')

        # optimizer
        self.parser.add_argument('--solver', type=str, default="ADAM", choices=["SGD", "ADAIM"], help="optimizer")
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for ADAM')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
        self.parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
        self.parser.add_argument('--loss', type=str, default="L1", help="optimizer [Options: SGD, ADAM]")

        # others
        self.parser.add_argument('--show_step', type=int, default=20, help='visdom show step')
        self.parser.add_argument('--save_epoch', type=int, default=1, help='checkpoints save_epoch')
        self.parser.add_argument('--max_epoch', type=int, default=2000, help='max epochs')
        self.parser.add_argument('--save_iteration', type=int, default=10000, help='checkpoints save iteration')
        self.parser.add_argument('--save_img_iteration', type=int, default=5000, help=' generated fake face image save iteration')
        self.parser.add_argument('--display_info_step', type=int, default=5, help='print info step')
