import json
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import cv2
import numpy as np
import torchvision
from torch.utils.data import Dataset
from PIL import Image as pil_image
import glob
from paths import Path
from .transform import xception_default_data_transforms


class FFPPDataset(Dataset):
    def __init__(self, opts, phase):
        self.root_img_dir, self.root_video_dir, self.inner_test_dir, self.output_dir = Path.db_dir('ff++')
        self.opts = opts
        self.phase = phase
        self.cuda = True if self.opts.device else False
        self.device = self.parse_str(self.opts.device)
        self.no_pair = self.opts.no_pair
        dataset_name = self.opts.dataset_name
        compression = self.opts.compression
        self.DATASET_PATHS = {
            'original': 'original_sequences/youtube',
            'Deepfakes': 'manipulated_sequences/Deepfakes',
            'Face2Face': 'manipulated_sequences/Face2Face',
            'FaceSwap': 'manipulated_sequences/FaceSwap',
            'NeuralTextures': 'manipulated_sequences/NeuralTextures'
        }
        # self.COMPRESSION = ['c0', 'c23', 'c40']
        self.splits_json_dir = './dataset/splits'
        self.imgs_dir = os.path.join(self.root_img_dir, self.DATASET_PATHS[dataset_name], compression, 'face')
        self.anns_dir = os.path.join(self.root_img_dir, self.DATASET_PATHS[dataset_name], compression, 'anno')
        self.ori_img_dir = os.path.join(self.root_img_dir, self.DATASET_PATHS["original"], compression, 'face')
        self.ori_anns_dir = os.path.join(self.root_img_dir, self.DATASET_PATHS["original"], compression, 'anno')

        self.imgsA_path = []  # real
        self.imgsB_path = []  # fake
        self.annsA_path = []
        self.annsB_path = []

        self.get_split_path()

        if self.no_pair:
            self.imgs_path = self.imgsA_path + self.imgsB_path
            self.anns_path = self.annsA_path + self.annsB_path
            self.labels = list(np.ones(len(self.imgsA_path))) + list(np.zeros(len(self.imgsB_path)))
            assert (len(self.imgs_path) == len(self.labels)), "Length of img and labels do not match"

    def __getitem__(self, index):
        if self.no_pair:
            img_path = self.imgs_path[index]
            ann_path = self.anns_path[index]
            label = np.array(self.labels[index])
            image = cv2.imread(img_path)
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)

            # Revert from BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ann_data = np.array(ann_data) / 256  # normalize the points

            # Preprocess using the preprocessing function used during training and
            # casting it to PIL image
            if not self.opts.keepOri:
                preprocess = xception_default_data_transforms[self.phase]
                preprocessed_image = preprocess(pil_image.fromarray(image))
            else: # data type ->double
                preprocessed_image = torch.from_numpy((image / 255.0).astype(np.float64))
                preprocessed_image = preprocessed_image.permute(2, 0, 1)
            ann_data = torch.from_numpy(ann_data)

            return preprocessed_image, ann_data, torch.from_numpy(label)
        else:
            img_real_path = self.imgsA_path[index]
            img_fake_path = self.imgsB_path[index]
            ann_real_path = self.annsA_path[index]
            ann_fake_path = self.annsB_path[index]
            img_real = cv2.imread(img_real_path)
            img_fake = cv2.imread(img_fake_path)
            with open(ann_fake_path) as f:
                ann_fake_data = json.load(f)
            with open(ann_real_path) as f:
                ann_real_data = json.load(f)
            img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
            img_fake = cv2.cvtColor(img_fake, cv2.COLOR_BGR2RGB)
            ann_fake_data = np.array(ann_fake_data) / 256.0  # normalize the points
            ann_real_data = np.array(ann_real_data) / 256.0  # normalize the points

            if not self.opts.keepOri:
                preprocess = xception_default_data_transforms[self.phase]
                img_real = preprocess(pil_image.fromarray(img_real))
                img_fake = preprocess(pil_image.fromarray(img_fake))

            else:
                img_real = torch.from_numpy(img_real / 255.0)
                img_fake = torch.from_numpy(img_fake / 255.0)
            img_real = img_real.permute(2, 0, 1)
            img_fake = img_fake.permute(2, 0, 1)
            ann_fake_data = torch.from_numpy(ann_fake_data)
            ann_real_data = torch.from_numpy(ann_real_data)
            return img_real, img_fake, ann_real_data, ann_fake_data



    def __len__(self):
        if self.opts.no_pair:
            return len(self.imgs_path)
        else:
            return len(self.imgsA_path)

    def get_split_path(self):
        json_phase_path = os.path.join(self.splits_json_dir, self.phase+".json")
        with open(json_phase_path, 'r') as f:
            json_data = json.load(f)
        for itm in json_data:
            # face in target video is replaced by a face that has been observed in a source video or image.
            target_id, src_id = itm
            fake_seq_name = str(target_id) + "_" + str(src_id)
            real_seq_name = str(target_id)
            fake_seqs = glob.glob(os.path.join(self.imgs_dir, fake_seq_name, "*.png"))
            fake_jsons = glob.glob(os.path.join(self.anns_dir, fake_seq_name, "*.json"))
            real_seqs = glob.glob(os.path.join(self.ori_img_dir, real_seq_name, "*.png"))
            real_jsons = glob.glob(os.path.join(self.ori_anns_dir, real_seq_name, "*.json"))
            self.imgsA_path += real_seqs
            self.annsA_path += real_jsons
            self.imgsB_path += fake_seqs
            self.annsB_path += fake_jsons

    def parse_str(self, ids):
        str_ids = ids.split(',')
        ids_list = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                ids_list.append(id)
        return ids_list


def tensor2img(img_t):
    if len(img_t.size()) == 4:
        img = img_t[0].detach().to("cpu").numpy()
        img = np.transpose(img, (1, 2, 0))
    elif len(img_t.size()) == 3:
        img = img_t.detach().to("cpu").numpy()
        img = np.transpose(img, (1, 2, 0))

    return img


def draw_batch(opts, batch_imgs, batch_anns, labels, count):
    NUM = batch_imgs.shape[0]
    rtn_size = (batch_imgs.shape[0], batch_imgs.shape[1], batch_imgs.shape[2], batch_imgs.shape[3])
    rtn = torch.FloatTensor(torch.Size(rtn_size)).zero_()
    batch_imgs = np.array(batch_imgs).transpose(0, 2, 3, 1)
    # print(batch_imgs.shape)
    for slice_id in range(NUM):
        img = batch_imgs[slice_id].squeeze()
        if not opts.keepOri:
            img = img * 0.5 + 0.5
        img = img * 255
        img = img.astype(np.uint8)
        t = img.copy()
        ann = batch_anns[slice_id].squeeze()
        ann = ann * img.shape[1]
        # print(img.shape[0])
        for p in ann:
            cv2.circle(t, tuple(p), color=(0,0,255), radius=2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if labels[slice_id] == 1:
            cv2.putText(t, 'real', fontFace=font, fontScale=1.2, org=(50,50), color=(255,255,0))
        else:
            cv2.putText(t, 'fake', fontFace=font, fontScale=1.2, org=(50,50), color=(255,255,0))
        rtn[slice_id, :, :, :] = torch.from_numpy(t.transpose(2, 0, 1))

    rtn = torchvision.utils.make_grid(rtn, nrow=4)
    rtn = tensor2img(rtn)
    rtn = cv2.cvtColor(rtn, cv2.COLOR_RGB2BGR)
    if not os.path.exists("./insight_results"):
        os.mkdir("./insight_results/")
    cv2.imwrite('./insight_results/' + str(count) + ".jpg", rtn)
    return rtn


def _main(opts):
    train_data = FFPPDataset(opts)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    count = 0
    for data in tqdm(train_loader):
        if opts.no_pair:
            preprocessed_image, ann_data, labels = data
            draw_batch(opts, preprocessed_image, ann_data, labels, count)
        else:
            img_real, img_fake, ann_real_data, ann_fake_data = data
            draw_batch(opts, img_real, ann_real_data, count)
            count += 1
            draw_batch(opts, img_fake, ann_fake_data, count)
        count += 1




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepFake-Pytorch')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                        help='number of epochs to train (default: 10000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--phase', type=str, default="train",
                        help='phase')
    parser.add_argument('--device', type=str, default="0", help='use device ex. "0,1,2"')
    parser.add_argument('--no_pair', action='store_false',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset_name', type=str, default="Deepfakes",
                        help='choose from Deepfakes, Face2Face, FaceSwap, NeuralTextures')
    parser.add_argument('--compression', type=str, default="c23", help='choose from raw, c23, c40')
    parser.add_argument('--keepOri', action='store_false', help='debug, no norm')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    _main(args)