import sys
sys.path.append('/mnt/hdd2/std/zyf/dfke/faceGen/FaceShifter-pytorch/datasets')
sys.path.append('.')
from path import Path
from BaseDataset import BaseDataset
import random
import numpy as np
import os
import cv2


class FaceForensicsDataset(BaseDataset):

    def __init__(self, draw_landmarks):
        BaseDataset.__init__(self, draw_landmarks=draw_landmarks)

    def parseImagePath(self, image_path):
        """
        set the path.
        self.frame_path = ""
        self.iden_frame_path = ""
        self.dif_frame_path = ""
        self.frame_anno_path = ""
        self.iden_frame_anno_path = ""
        self.dif_frame_anoo_path = ""

        e.g. /mnt/disk3/std/zyf/dataset/deepfake/faceforensics++/original_sequences/youtube/c23/face/999/0334.png
        TODO 定位到每个目录，然后随机。
        :param image_path:
        :return:
        """
        self.frame_path = image_path
        video_id, frame_name = image_path.split('/')[-2:]
        frame_id = frame_name.split('.')[0]
        anno_dir_path = os.path.join(self.root_anno_dir, video_id)
        self.frame_anno_path = os.path.join(anno_dir_path, frame_id+'.json')

        iden_frame_name = frame_name
        video_frames_path = os.path.join(self.root_face_dir, video_id)
        video_frames_list = os.listdir(video_frames_path)
        while iden_frame_name == frame_name:
            iden_frame_name = random.choice(video_frames_list)
        self.iden_frame_path = os.path.join(video_frames_path, iden_frame_name)
        self.iden_frame_anno_path = os.path.join(anno_dir_path, iden_frame_name.split('.')[0]+'.json')

        idens_list = os.listdir(self.root_face_dir)
        dif_video_id = video_id
        while dif_video_id == video_id:
            dif_video_id = random.choice(idens_list)
        dif_frames_path = os.path.join(self.root_face_dir, dif_video_id)
        dif_frames_list = os.listdir(dif_frames_path)
        dif_frame_name = self.chooseDifIdenFace(dif_frames_list)
        self.dif_frame_path = os.path.join(dif_frames_path, dif_frame_name)
        self.dif_frame_anoo_path = os.path.join(self.root_anno_dir, dif_video_id, dif_frame_name.split('.')[0] + '.json')

    def chooseDifIdenFace(self, dif_frames_list):
        """
        get the dif identy face from the dif_frames_list according to the set method.
        here
            1. random choose one
            2. random choose 10 or stm. then using the ls2 to choose the most match x,y landmarks.
        :param dif_frames_list:
        :return:
        """
        return random.choice(dif_frames_list)

    def initiate(self):
        """
        get the dataset.
        :return:
        """
        self.root_face_dir, \
        self.root_anno_dir, \
        self.inner_test_dir, \
        self.output_dir  = Path.FaceForensicsPaths(6665)
        for video_id in os.listdir(self.root_face_dir):
            video_path = os.path.join(self.root_face_dir, video_id)
            for frame_id in os.listdir(video_path):
                self.dataset.append(os.path.join(video_path, frame_id))

def get_grid_image(X):
    X = X[:2]
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]) * 0.5 + 0.5
    return X


def make_image(Xs, Xt, Y):
    Xs = get_grid_image(Xs)
    Xt = get_grid_image(Xt)
    Y = get_grid_image(Y)
    return torch.cat((Xs, Xt, Y), dim=1).numpy()

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torch
    import torchvision
    dataset = FaceForensicsDataset(draw_landmarks=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=True)
    for iteration, data in enumerate(dataloader):
        faces, landmarks = data
        face, iden_face, dif_face = faces
        # anno, iden_anno, dif_anno = landmarks
        img = make_image(face, iden_face, dif_face)
        cv2.imwrite('./test.jpg', img.transpose([1, 2, 0]) * 255)



