import torch
import numpy as np
import os
import random
import main
from torch.utils.data import Dataset
import PIL
from PIL import Image
import matplotlib.pyplot as plt


class AlexNetDataset(Dataset):
    img_w, img_h = 225, 225
    dir_name_trim_length = -9
    random_crop_ratio = 0.2

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        file_name = item['file_path']
        self.augmentation = item['augmentation']
        self.file_path = os.path.join(main.lfw_dataset_path, file_name[:self.dir_name_trim_length], file_name)
        self.crops = item['crops'].split(' ')
        self.crops = list(map(float, self.crops))
        self.landmarks = item['landmarks'].split(' ')
        self.landmarks = list(map(float, self.landmarks))

        # Process random cropping.
        if self.augmentation[1] == '1':
            shift_coordinates = np.array([0.0, 0.0])
            shift = (self.crops[2] - self.crops[0]) * self.random_crop_ratio
            shift = random.uniform(0, shift)
            shift *= [-1, 1][random.randrange(2)]
            shift_coordinates[0] = shift
            shift = (self.crops[2] - self.crops[0]) * self.random_crop_ratio
            shift = random.uniform(0, shift)
            shift *= [-1, 1][random.randrange(2)]
            shift_coordinates[1] = shift
            self.crops += np.tile(shift_coordinates, int(len(self.crops) / 2))

        # Prepare image tensor.
        img_tensor = Image.open(self.file_path)
        img_tensor = self.crop(img_tensor)
        img_tensor = self.flip(img_tensor)
        img_tensor = self.brighten(img_tensor)
        img_tensor = self.normalize(img_tensor)

        # Prepare landmark tensor.
        landmarks = np.asarray(self.landmarks)
        landmarks = landmarks.reshape(7, 2)
        landmarks = self.crop(landmarks)
        landmark_tensors = torch.from_numpy(landmarks)

        # 225 x 225 x 3 input image tensor. 7 * 2 landmark tensor.
        return img_tensor, landmark_tensors

    def crop(self, _input):
        # Case that a image need to be cropped.
        if isinstance(_input, PIL.JpegImagePlugin.JpegImageFile):
            img = _input
            img = img.crop(self.crops)
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
        ratio = float(self.img_w) / (self.crops[2] - self.crops[0])
        return np.subtract(landmarks, shift) * ratio

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

        target = self[idx]
        image = target[0]
        landmarks = target[1]
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
