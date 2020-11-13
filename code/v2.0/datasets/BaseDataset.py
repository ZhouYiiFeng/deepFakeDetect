from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from utils.face_align_tools import align_multi
from utils.face_align_tools import get_reference_facial_points
import cv2
import numpy as np
from abc import ABCMeta, abstractmethod


class BaseDataset(Dataset):
    def __init__(self, draw_landmarks):
        self.refrence = get_reference_facial_points(default_square=True)
        self.draw_landmarks = draw_landmarks
        self.dataset = []
        self.frame_path = ""
        self.iden_frame_path = ""
        self.dif_frame_path = ""
        self.frame_anno_path = ""
        self.iden_frame_anno_path = ""
        self.dif_frame_anoo_path = ""
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.initiate()

    @abstractmethod
    def initiate(self):
        """
        get the dataset.
        :return:
        """
        pass

    @abstractmethod
    def parseImagePath(self, image_path):
        """
        set the path.
        self.frame_path = ""
        self.iden_frame_path = ""
        self.dif_frame_path = ""
        self.frame_anno_path = ""
        self.iden_frame_anno_path = ""
        self.dif_frame_anoo_path = ""

        :param image_path:
        :return:
        """
        pass

    def load_ann(self, anno_root_dir):
        try:
            import json
            with open(anno_root_dir, 'r') as f:
                ann_data = json.load(f)
            ann_data = np.array(ann_data)  # normalize the points
        except Exception as e:
            print("Load json error")
            return []

        # ann_data = torch.from_numpy(ann_data)
        return ann_data

    def __getitem__(self, index):
        image_path = self.dataset[index]

        # frame_path, iden_frame_path, \
        # dif_frame_path, frame_anno_path, \
        # iden_frame_anno_path, dif_frame_anoo_path =

        self.parseImagePath(image_path)

        Xs = cv2.imread(self.frame_path)
        Xi = cv2.imread(self.iden_frame_path)
        Xd = cv2.imread(self.dif_frame_path)

        frames = [Xs, Xi, Xd]
        annos = [self.frame_anno_path, self.iden_frame_anno_path, self.dif_frame_anoo_path]
        faces = []
        landmarks = []
        for img, anno in zip(frames, annos):
            landmark = self.load_ann(anno)
            face, landmark_new = align_multi(Image.fromarray(img), landmark, self.refrence, crop_size=(256, 256), draw_landmarks=self.draw_landmarks)
            faces.append(self.transforms(face))
            landmarks.append(landmark_new)
        return faces, landmarks

    def __len__(self):
        return len(self.dataset)


