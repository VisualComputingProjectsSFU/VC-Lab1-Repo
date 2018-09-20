import torch
import numpy as np
import os
import main
from torch.utils.data import Dataset
import PIL
from PIL import Image


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
        self.crops = item['crops'].split(' ')
        self.landmarks = item['landmarks'].split(' ')

        # Prepare image tensor.
        img_tensor = Image.open(self.file_path)
        img_tensor = self.crop(img_tensor)
        img_tensor = self.flip(img_tensor)
        img_tensor = self.brighten(img_tensor)
        img_tensor = self.normalize(img_tensor)

        # Prepare landmark tensor.
        landmarks = np.asarray(list(map(float, self.landmarks)))
        landmarks = landmarks.reshape(7, 2)
        landmarks = self.crop(landmarks)
        landmark_tensors = torch.from_numpy(landmarks)

        # 225 x 225 x 3 input image tensor. 7 * 2 landmark tensor.
        return img_tensor, landmark_tensors

    def crop(self, _input):
        # Case that a image need to be cropped.
        if isinstance(_input, PIL.JpegImagePlugin.JpegImageFile):
            img = _input
            img = img.crop(list(map(int, self.crops)))
            img = img.resize((self.img_w, self.img_h), PIL.Image.ANTIALIAS)
            img = np.asarray(img)

            h, w = img.shape[0], img.shape[1]
            chanel = img.shape[2]

            # Create image tensor
            img_tensor = torch.from_numpy(img)

            # Reshape to (3, 225, 225), the 1 is the channel size
            img_tensor = img_tensor.view((chanel, h, w))
            return img_tensor

        # Case that the landmarks need to be adjusted after cropping.
        landmarks = _input
        shift = np.array([[self.crops[0], self.crops[1]]])
        shift = np.repeat(shift, 7, axis=0)
        ratio = float(self.img_w) / (float(self.crops[2]) - float(self.crops[0]))
        return np.subtract(landmarks.astype(float), shift.astype(float)) * ratio

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
