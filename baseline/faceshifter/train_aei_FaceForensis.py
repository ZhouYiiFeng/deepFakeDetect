from network.MultiScaleDiscriminator import *
from torch.utils.data import DataLoader
from face_modules.model import Backbone
from datasets.Faceforensis_Dataset import FaceForensicsDataset
import torch.nn.functional as F
import torch.optim as optim
from network.aei import *
from apex import amp
import torchvision
import visdom
import torch
import time
import cv2
import os
import sys
sys.path.append('./datasets')
sys.path.append('.')

gpu_id = 2
os.environ['CUDA_VISIBLE_DEVICES'] = "%d" % gpu_id
vis = visdom.Visdom(server='127.0.0.1', env='se_fs', port=8097) # 8097
batch_size = 8
lr_G = 4e-4
lr_D = 4e-4
max_epoch = 2000
show_step = 20  # visdom show step
save_epoch = 1  # checkpoints save_epoch
display_info_step = 5  # info display step
model_save_path = './checkpoints/'
optim_level = 'O1'
inner_path = "./mid_results"
model_name = "Test02FF++"
load_pretrian_model = False
use_apex = False


checkponits_dir = os.path.join(model_save_path, model_name)
inner_dir = os.path.join(inner_path, model_name)
if not os.path.exists(checkponits_dir):
    os.makedirs(checkponits_dir)
if not os.path.exists(inner_dir):
    os.makedirs(inner_dir)

device = torch.device('cuda:%d' % gpu_id)

G = AEI_Net(c_id=512).to(device)
D = MultiscaleDiscriminator(input_nc=3, n_layers=6, norm_layer=torch.nn.InstanceNorm2d).to(device)
G.train()
D.train()

arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(torch.load('./saved_models/model_ir_se50.pth', map_location=device), strict=False)

opt_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr_D, betas=(0, 0.999))

if use_apex:
    G, opt_G = amp.initialize(G, opt_G, opt_level=optim_level)
    D, opt_D = amp.initialize(D, opt_D, opt_level=optim_level)


if load_pretrian_model:
    try:
        G.load_state_dict(torch.load('./saved_models/G_latest.pth', map_location=torch.device('cpu')), strict=False)
        D.load_state_dict(torch.load('./saved_models/D_latest.pth', map_location=torch.device('cpu')), strict=False)
    except Exception as e:
        print(e)

dataset = FaceForensicsDataset(draw_landmarks=False)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


MSE = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()


def hinge_loss(X, positive=True):
    if positive:
        return torch.relu(1-X).mean()
    else:
        return torch.relu(X+1).mean()


def get_grid_image(X):
    X = X[:8]
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]) * 0.5 + 0.5
    return X


def make_image(Xs, Xt, Y):
    Xs = get_grid_image(Xs)
    Xt = get_grid_image(Xt)
    Y = get_grid_image(Y)
    return torch.cat((Xs, Xt, Y), dim=1)

# print(torch.backends.cudnn.benchmark)
inner_count = 0 # save mid results idx. -> idx.jpg

for epoch in range(0, max_epoch):
    for iteration, data in enumerate(dataloader):
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
            Xs = Xs.to(device)
            Xt = Xt.to(device)
            with torch.no_grad():
                embed, Xs_feats = arcface(F.interpolate(Xs, [112, 112], mode='bilinear', align_corners=True))
            same_person = same_person.to(device)

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

            Y_attr = G.get_attr(Y)
            L_attr = 0
            for i in range(len(Xt_attr)):
                L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_attr[i], 2).reshape(batch_size, -1), dim=1).mean()
            L_attr /= 2.0
            L_rec = 0.5 * MSE(Y, Xt) * same_person
            # L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y - Xt, 2).reshape(batch_size, -1), dim=1) * same_person)

            l_adv = 1
            l_att = 10
            l_id = 10
            l_rec = 5

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

        if len(image_show) != 0:
            image = torch.cat((image_show[0], image_show[1], image_show[2]), dim=1).numpy()
            vis.image(image[::-1, :, :], opts={'title': 'result'}, win='result')
            cv2.imwrite(inner_dir + '/latest.jpg', image.transpose([1, 2, 0]) * 255)
            if iteration % save_epoch == 0:
                cv2.imwrite(inner_dir + '/%s.jpg' % (str(inner_count)), image.transpose([1, 2, 0]) * 255)
                inner_count += 1

        if iteration % save_epoch == 0:
            # image = make_image(Xs, Xt, Y)
            # cv2.imwrite(inner_dir + '/%s.jpg' % (str(inner_count)), image.transpose([1, 2, 0])*255)
            # inner_count += 1
            torch.save(G.state_dict(), checkponits_dir + '/G_latest.pth')
            torch.save(D.state_dict(), checkponits_dir + '/D_latest.pth')

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
