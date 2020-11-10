#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
---------------------------------
@ Author : JoeyF.Zhou           -
@ Home Page : www.zhoef.com     -
@ From : UESTC                  - 
---------------------------------
@ Date : 11/6/2020 7:43 PM
@ Project Name : FaceShifter-pytorch-master
@ Description :
    This code is writen by joey.
    @ function:
        Using DataParallel
    
"""
from network.MultiScaleDiscriminator import *
from torch.utils.data import DataLoader
from face_modules.model import Backbone
from datasets.Faceforensis_Dataset import FaceForensicsDataset
import torch.nn.functional as F
import torch.optim as optim
from network.my_aei import *
from apex import amp
import torchvision
import visdom
import torch
import time
import cv2
import os
import sys
from utils.util import *
sys.path.append('./datasets')
sys.path.append('.')

gpu_id = 0
gpu_id2 = 2
os.environ['CUDA_VISIBLE_DEVICES'] = "0,2"
batch_size = 2
lr_G = 4e-4
lr_D = 4e-4
max_epoch = 2000
show_step = 20  # visdom show step
save_epoch = 1  # checkpoints save_epoch
save_iteration = 10000  # checkpoints save iteration
save_img_iteration = 5000  # generated fake face image save iteration
display_info_step = 5  # info display step
model_save_path = './checkpoints/'
optim_level = 'O1'
inner_path = "./mid_results"
model_name = "my_aei"
vis = visdom.Visdom(server='127.0.0.1', env=model_name, port=8097) # 8097
load_pretrian_model = False
use_apex = False


checkponits_dir = os.path.join(model_save_path, model_name)
inner_dir = os.path.join(inner_path, model_name)
if not os.path.exists(checkponits_dir):
    os.makedirs(checkponits_dir)
if not os.path.exists(inner_dir):
    os.makedirs(inner_dir)

device = torch.device('cuda:%d' % gpu_id2)
device2 = torch.device('cuda:%d' % gpu_id)

G = AETNet().to(device)
D = MultiscaleDiscriminator(input_nc=3, n_layers=6, norm_layer=torch.nn.InstanceNorm2d).to(device2)
G = torch.nn.DataParallel(G)
D = torch.nn.DataParallel(D)
G.train()
D.train()

arcface = Backbone(50, 0.6, 'ir_se').to(device2)
arcface.eval()
arcface.load_state_dict(torch.load('./saved_models/model_ir_se50.pth', map_location=device2), strict=False)

opt_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr_D, betas=(0, 0.999))

if use_apex:
    G, opt_G = amp.initialize(G, opt_G, opt_level=optim_level)
    D, opt_D = amp.initialize(D, opt_D, opt_level=optim_level)

resume_epoch = 0
resume_iter = 0
l_adv = 1
l_att = 10
l_id = 10
l_rec = 5
if load_pretrian_model:
    checkpoint = torch.load(os.path.join(checkponits_dir,"model_latest.pth"))
    G.load_state_dict(checkpoint['Gnet'])
    D.load_state_dict(checkpoint['Dnet'])
    opt_G.load_state_dict(checkpoint['optimizerG'])
    opt_D.load_state_dict(checkpoint['optimizerD'])
    resume_epoch = checkpoint['epoch']
    resume_iter = checkpoint['iter']
    params = torch.load(os.path.join(checkponits_dir,"params.pth"))
    l_adv = params['l_adv']
    l_att = params['l_att']
    l_id = params['l_id']
    l_rec = params['l_rec']

dataset = FaceForensicsDataset(draw_landmarks=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

MSE = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()

# print(torch.backends.cudnn.benchmark)
inner_count = 0 # save mid results idx. -> idx.jpg

for epoch in range(resume_epoch, max_epoch):
    for iteration, data in enumerate(dataloader, resume_iter):
        start_time = time.time()
        faces, landmarks = data
        face, iden_face, dif_face = faces
        image_show = []
        lossD_all = 0.0
        lossG_all = 0.0
        L_adv_all = 0.0
        L_id_all = 0.0
        L_attr_all = 0.0
        L_rec_all = 0.0
        for Xs, Xt, same_person in [[face, face, torch.FloatTensor([1])],
                                    [face, iden_face, torch.FloatTensor([0.5])],
                                    [face, dif_face, torch.FloatTensor([0])]]:
            # Xs, Xt, same_person = data
            Xs = Xs.to(device2)
            Xt = Xt.to(device)
            with torch.no_grad():
                embed, Xs_feats = arcface(F.interpolate(Xs, [112, 112], mode='bilinear', align_corners=True))
            same_person = same_person.to(device2)
            embed = embed.to(device)

            # train G
            opt_G.zero_grad()
            Y, Xt_attr = G(Xt, embed)
            Y = Y.to(device2)
            Xt_attr = Xt_attr.to(device2)
            Di = D(Y)
            L_adv = 0

            for di in Di:
                L_adv += hinge_loss(di[0], True)

            Y_aligned = Y
            ZY, Y_feats = arcface(F.interpolate(Y_aligned, [112, 112], mode='bilinear', align_corners=True))
            L_id = (1 - torch.cosine_similarity(embed, ZY, dim=1)).mean()

            Y_attr = G.getAttributes(Y)
            L_attr = 0
            for i in range(len(Xt_attr)):
                L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_attr[i], 2).reshape(batch_size, -1), dim=1).mean()
            L_attr /= 2.0
            L_rec = 0.5 * MSE(Y, Xt) * same_person
            # L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y - Xt, 2).reshape(batch_size, -1), dim=1) * same_person)

            lossG = l_adv*L_adv + l_att*L_attr + l_id*L_id + l_rec*L_rec

            if use_apex:
                with amp.scale_loss(lossG, opt_G) as scaled_loss:
                    scaled_loss.backward()
            else:
                lossG.backward()
            opt_G.step()
            lossG_all += lossG.item()
            L_adv_all += L_adv.item()
            L_attr_all += L_attr.item()
            L_id_all += L_id.item()
            L_rec_all += L_rec.item()
            # train D
            opt_D.zero_grad()
            fake_D = D(Y.detach())
            loss_fake = 0
            for di in fake_D:
                loss_fake += hinge_loss(di[0], False)

            true_D = D(Xs)
            loss_true = 0
            for di in true_D:
                loss_true += hinge_loss(di[0], True)

            lossD = 0.5*(loss_true.mean() + loss_fake.mean())
            lossD_all += lossD.item()
            if use_apex:
                with amp.scale_loss(lossD, opt_D) as scaled_loss:
                    scaled_loss.backward()
            else:
                lossD.backward()
            opt_D.step()

            batch_time = time.time() - start_time

            if iteration % show_step == 0:
                image = make_image(Xs, Xt, Y)
                image_show.append(image)

            Xs = Xs.to(device)
            Xt = Xt.to(device)

        if len(image_show) != 0:
            image = torch.cat((image_show[0], image_show[1], image_show[2]), dim=1).numpy()
            vis.image(image[::-1, :, :], opts={'title': 'result'}, win='result')
            cv2.imwrite(inner_dir + '/latest.jpg', image.transpose([1, 2, 0]) * 255)
            if iteration % save_img_iteration == 0:
                cv2.imwrite(inner_dir + '/%s.jpg' % (str(inner_count)), image.transpose([1, 2, 0]) * 255)
                inner_count += 1

        if iteration % save_iteration == 0:
            """
            checkpoint = torch.load(os.path.join(checkponits_dir,"model_latest.pth"))
            G.load_state_dict(checkpoint['Gnet'])
            D.load_state_dict(checkpoint['Dnet'])
            opt_G.load_state_dict(checkpoint['optimizerG'])
            opt_D.load_state_dict(checkpoint['optimizerD'])
            resume_epoch = checkpoint['epoch']
            resume_iter = checkpoint['iter']
            """
            state = {
                'Gnet': G.state_dict(),
                'Dnet': D.state_dict(),
                'optimizerG': opt_G.state_dict(),
                'optimizerD': opt_D.state_dict(),
                'epoch': epoch,
                'iter': iteration
            }
            params_state = {
                'l_adv': l_adv,
                'l_att': l_att,
                'l_id': l_id,
                'l_rec': l_rec
            }
            torch.save(state, checkponits_dir + '/model_latest.pth')
            torch.save(params_state, checkponits_dir + '/params.pth')
            torch.save(state, checkponits_dir + '/ep_%s_iter_%s_.pth' % (epoch, iteration))

        if iteration % display_info_step == 0:
            info = "Train: [GPU %s], Batch_time: %s \n" % (device, str(batch_time))
            info += "Train: Epoch %d; Batch %d / %d: \n" % (epoch, iteration, len(dataloader))
            info += "G_lr = %s; " % (str(lr_G))
            info += "D_lr = %s; \n" % (str(lr_D))
            info += "\t\t%25s = %f\n" % ("lossD_all", lossD_all)
            info += "\t\t%25s = %f\n" % ("lossG_all", lossG_all)
            info += "\t\t%25s = %f\n" % ("L_adv_all",  L_adv_all)
            info += "\t\t%25s = %f\n" % ("L_id_all",   L_id_all)
            info += "\t\t%25s = %f\n" % ("L_attr_all", L_attr_all)
            info += "\t\t%25s = %f\n" % ("L_rec_all",  L_rec_all)
            print(info)

