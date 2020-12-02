#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
---------------------------------
@ Author : JoeyF.Zhou           -
@ Home Page : www.zhoef.com     -
@ From : UESTC                  - 
---------------------------------
@ Date : 11/11/2020 4:44 PM
@ Project Name : FaceShifter-pytorch-master
@ Description :
    This code is writen by joey.
    @ function:
    
"""
import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="TDMSNet")
        self.initialized = False

    def initialize(self):

        # data
        self.parser.add_argument('--dataset', type=str, default='FaceForensics', help='name of the dataset')
        self.parser.add_argument('--num_workers', type=int, default=8, help='number of threads for data loader to use')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='dir of checkpoints')

        # model
        self.parser.add_argument('--batch_size', type=int, default=2, help='training batch size')

        # others
        self.parser.add_argument('--exp_name', type=str, default='my_aei', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--cpu', action='store_true', help='use cpu?')
        self.parser.add_argument('--resume', action='store_true', help='resume the checkpoint')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--resume_epoch', type=int, default=0, help='resume epoch')
        self.parser.add_argument('--resume_iter', type=int, default=0, help='resume iter')
        self.parser.add_argument('--inner_path', type=str, default="./mid_results", help='path to store the inner results')
        # self.parser.add_argument('--resume_iter', type=str, default='0', help='resume iter')
        self.parser.add_argument('--use_apex', action='store_true', help='resume the checkpoint')

        self.initialized = True

    def parse_str(self, ids):
        str_ids = ids.split(',')
        ids_list = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                ids_list.append(id)
        return ids_list

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        self.opt.gpu_ids = self.parse_str(self.opt.gpu_ids)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.exp_name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        if save:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        return self.opt