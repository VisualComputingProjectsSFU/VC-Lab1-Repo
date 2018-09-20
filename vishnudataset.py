import torch
import numpy as np
import os
import random
import main
from torch.utils.data import Dataset
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from random import randint


class AlexNetDataset(Dataset):
    img_w, img_h = 225, 225
    dir_name_trim_length = -9

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        file_name = item['file_path']
        self.file_path = os.path.join(main.lfw_dataset_path, file_name[:self.dir_name_trim_length], file_name)
        self.crops = np.asarray(item['crops'])
        self.crops = self.crops.astype(float)
        self.landmarks = item['landmarks']
        self.landmarks = self.landmarks.astype(float)
        self.aug_types = item['aug_types']

        self.aug_type_list = list(self.aug_types)

        # Prepare image tensor.
        img = Image.open(self.file_path)

        if self.aug_type_list[0] == '1':
            img, self.landmarks = self.crop(img, self.landmarks, True)
        else:
            img, self.landmarks = self.crop(img, self.landmarks, False)

        if self.aug_type_list[1] == '1':
            img, self.landmarks = self.flip(img, self.landmarks)

        if self.aug_type_list[2] == '1':
            img, self.landmarks = self.brighten(img)

        # 225 x 225 x 3 input image tensor. 7 * 2 landmark tensor.
        return img_tensor, landmark_tensors

    def crop(self, image, landmarks, offset):
        # Case that a image need to be cropped.
        crops_offset = crops
        if offset == True:
            for index in range(0, 3):
                rand_offset = randint(-5, 5)
                crops_offset[index] = self.crops[index] + rand_offset
        else:
            crops_offset = self.crops

        img = image
        img = img.crop((crops_offset))
        features_offset = [crops_offset[0], crops_offset[1]]
        features_offset = np.tile(features_offset, (len(landmarks) / 2))
        cropped_landmarks = landmarks - features_offset
        return img, cropped_landmarks


@staticmethod
def flip(cropped_img):
    img = cropped_img
    return img


@staticmethod
def brighten(cropped_img):
    img = cropped_img
    return img


@staticmethod
def normalize(cropped_img):
    img = cropped_img
    return img


def preview(self, idx=-1, is_landmarks_displayed=True):
    if idx == -1:
        idx = random.randint(0, len(self))

    image = self[idx][0]
    landmarks = self[idx][1]
    print('Image tensor shape (C, H, W):', image.shape)
    print('Label tensor shape (X, Y):', landmarks.shape)

    channels = image.shape[0]
    h, w = image.shape[1], image.shape[2]

    nd_img = image.cpu().numpy()
    plt.figure(num='Preview')
    plt.imshow(nd_img.reshape(h, w, channels))

    if is_landmarks_displayed:
        nd_landmarks = landmarks.cpu().numpy()
        for i in range(0, len(nd_landmarks)):
            plt.plot(nd_landmarks[i][0], nd_landmarks[i][1:], 'bo')

    plt.xlim(0, 225)
    plt.ylim(225, 0)
    plt.title('Preview at Index [' + str(idx) + ']')
    plt.draw()
