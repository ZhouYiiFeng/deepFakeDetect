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
    my_train_v + model . + others.
    the param setting could be seen in ./scripts/v + model . + others . + exp_Times.
    @ function:
    Options + save G, D , params, separately.

"""
import torch
from network.MultiScaleDiscriminator import *
from torch.utils.data import DataLoader
from face_modules.model import Backbone
from datasets.Faceforensis_Dataset import FaceForensicsDataset
import torch.nn.functional as F
import torch.optim as optim
from network.my_aei import *
import visdom
import time
import cv2
import os
from utils.util import *
from options.train_options import TrainOptions

opts = TrainOptions().parse()
vis = visdom.Visdom(server='127.0.0.1', env=opts.exp_name, port=8097) # 8097

checkponits_dir = os.path.join(opts.checkpoints_dir, opts.exp_name)
inner_dir = os.path.join(opts.inner_path, opts.exp_name)
if not os.path.exists(checkponits_dir):
    os.makedirs(checkponits_dir)
if not os.path.exists(inner_dir):
    os.makedirs(inner_dir)

gpu_str = parse_gpu_id(opts.gpu_ids)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str

G = AETNet().cuda()
D = MultiscaleDiscriminator(input_nc=3, n_layers=6, norm_layer=torch.nn.InstanceNorm2d).cuda()

G.train()
D.train()

arcface = Backbone(50, 0.6, 'ir_se').cuda()
arcface.eval()
arcface.load_state_dict(torch.load('./saved_models/model_ir_se50.pth'), strict=False)

opt_G = optim.Adam(G.parameters(), lr=opts.lr_G, betas=(0, 0.999))
opt_D = optim.Adam(D.parameters(), lr=opts.lr_D, betas=(0, 0.999))

resume_epoch = 0
resume_iter = 0
l_adv = 1
l_att = 10
l_id = 5
l_rec = 10
inner_count = 0  # save mid results idx. -> idx.jpg
if opts.resume:
    G_checkpoint = torch.load(os.path.join(checkponits_dir, "G_latest.pth"))
    D_checkpoint = torch.load(os.path.join(checkponits_dir, "D_latest.pth"))
    param_checkpoint = torch.load(os.path.join(checkponits_dir, "params.pth"))
    G.load_state_dict(G_checkpoint)
    D.load_state_dict(D_checkpoint)
    # opt_G.load_state_dict(param_checkpoint['optimizerG'])
    # opt_D.load_state_dict(param_checkpoint['optimizerD'])
    resume_epoch = param_checkpoint['epoch']
    resume_iter = param_checkpoint['iter']
    l_adv = param_checkpoint['l_adv']
    l_att = param_checkpoint['l_att']
    l_id = param_checkpoint['l_id']
    l_rec = param_checkpoint['l_rec']
    inner_count = param_checkpoint['inner_count']

dataset = FaceForensicsDataset(draw_landmarks=False)
dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True)

MSE = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()

for epoch in range(resume_epoch, opts.max_epoch):
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
            Xs = Xs.cuda()
            Xt = Xt.cuda()
            same_person = same_person.cuda()
            with torch.no_grad():
                embed, Xs_feats = arcface(F.interpolate(Xs, [112, 112], mode='bilinear', align_corners=True))

            # train G
            opt_G.zero_grad()
            Y, Xt_attr = G(Xt, embed)

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
                L_attr += MSE(Xt_attr[i], Y_attr[i])
            L_attr /= 2.0
            L_rec = 0.5 * MSE(Y, Xt) * same_person
            # L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y - Xt, 2).reshape(batch_size, -1), dim=1) * same_person)

            lossG = l_adv * L_adv + l_att * L_attr + l_id * L_id + l_rec * L_rec

            if opts.use_apex:
                # with amp.scale_loss(lossG, opt_G) as scaled_loss:
                #     scaled_loss.backward()
                pass
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

            lossD = 0.5 * (loss_true.mean() + loss_fake.mean())
            lossD_all += lossD.item()
            if opts.use_apex:
                # with amp.scale_loss(lossD, opt_D) as scaled_loss:
                #     scaled_loss.backward()
                pass
            else:
                lossD.backward()
            opt_D.step()

            batch_time = time.time() - start_time

            if iteration % opts.show_step == 0:
                image = make_image(Xs, Xt, Y)
                image_show.append(image)

        if len(image_show) != 0:
            image = torch.cat((image_show[0], image_show[1], image_show[2]), dim=1).numpy()
            vis.image(image[::-1, :, :], opts={'title': 'result'}, win='result')
            cv2.imwrite(inner_dir + '/latest.jpg', image.transpose([1, 2, 0]) * 255)
            if iteration % opts.save_img_iteration == 0:
                cv2.imwrite(inner_dir + '/%s.jpg' % (str(inner_count)), image.transpose([1, 2, 0]) * 255)
                inner_count += 1

        if iteration % opts.save_iteration == 0:
            params_state = {
                'l_adv': l_adv,
                'l_att': l_att,
                'l_id': l_id,
                'l_rec': l_rec,
                'epoch': epoch,
                'iter': iteration,
                'inner_count': inner_count
                # 'optimizerG': opt_G.state_dict(),
                # 'optimizerD': opt_D.state_dict()
            }
            torch.save(G.state_dict(), checkponits_dir + '/G_latest.pth')
            torch.save(D.state_dict(), checkponits_dir + '/D_latest.pth')
            torch.save(params_state, checkponits_dir + '/params.pth')
            torch.save(G.state_dict(), checkponits_dir + '/G_latest_epoch%d_ite_%d.pth' % (epoch, iteration))
            torch.save(D.state_dict(), checkponits_dir + '/D_latest_epoch%d_ite_%d.pth' % (epoch, iteration))
            torch.save(params_state, checkponits_dir + '/params_epoch%d_ite_%d.pth' % (epoch, iteration))

        if iteration % opts.display_info_step == 0:
            info = "Exp: %s, Train: [GPU %s], Batch_time: %s \n" % (opts.exp_name, gpu_str, str(batch_time))
            info += "Train: Epoch %d; Batch %d / %d: \n" % (epoch, iteration, len(dataloader))
            info += "G_lr = %s; " % (str(opts.lr_G))
            info += "D_lr = %s; \n" % (str(opts.lr_D))
            info += "\t\t%25s = %f\n" % ("lossD_all", lossD_all)
            info += "\t\t%25s = %f\n" % ("lossG_all", lossG_all)
            info += "\t\t%25s = %f\n" % ("L_adv_all", L_adv_all)
            info += "\t\t%25s = %f\n" % ("L_id_all", L_id_all)
            info += "\t\t%25s = %f\n" % ("L_attr_all", L_attr_all)
            info += "\t\t%25s = %f\n" % ("L_rec_all", L_rec_all)
            print(info)

