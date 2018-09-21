import torch
import numpy as np
import os
import random
from torch.utils.data import Dataset
import PIL
import PIL.JpegImagePlugin
from PIL import Image
import matplotlib.pyplot as plt
import main


class AlexNetDataset(Dataset):
    img_w, img_h = 225, 225
    dir_name_trim_length = -9
    random_crop_ratio = 0.15
    random_brighten_ratio = 0.8
    _is_crop_updated = False

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

        # Prepare image array.
        images = self.crop(Image.open(self.file_path))
        images = self.flip(images)
        images = self.brighten(images)
        images = self.normalize(images)
        images = np.array(images)

        # Prepare landmark array.
        landmarks = np.asarray(self.landmarks)
        landmarks = landmarks.reshape(7, 2)
        landmarks = self.crop(landmarks)
        landmarks = self.flip(landmarks)
        landmarks = self.normalize(landmarks)

        # Create tensors and reshape them to proper sizes.
        img_tensor = torch.Tensor(images.astype(float))
        img_tensor = img_tensor.view((images.shape[2], images.shape[0], images.shape[1]))
        landmark_tensors = torch.Tensor(landmarks)

        # 225 x 225 x 3 input image tensor. 7 * 2 landmark tensor.
        return img_tensor, landmark_tensors

    def crop(self, _input):
        # Update random cropping once if the option is enabled.
        if (self.augmentation[0] == '1') and not self._is_crop_updated:
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
            self._is_crop_updated = True

        # Case that a image need to be cropped.
        if isinstance(_input, PIL.JpegImagePlugin.JpegImageFile):
            img = _input
            img = img.crop(self.crops)
            img = img.resize((self.img_w, self.img_h), PIL.Image.ANTIALIAS)

            return np.asarray(img)

        # Case that the landmarks need to be adjusted after cropping.
        landmarks = _input
        shift = np.array([[self.crops[0], self.crops[1]]])
        shift = np.repeat(shift, 7, axis=0)
        ratio = float(self.img_w) / (self.crops[2] - self.crops[0])
        return np.subtract(landmarks, shift) * ratio

    def flip(self, _input):
        # Flip the image.
        if self.augmentation[1] == '1' and (_input.shape == (225, 225, 3)):
            return np.fliplr(_input)

        # Flip the landmarks.
        if self.augmentation[1] == '1':
            new_landmarks = np.ndarray(shape=(7, 2), dtype=float)
            new_landmarks[0][0] = self.img_w - _input[3][0]
            new_landmarks[0][1] = _input[3][1]
            new_landmarks[1][0] = self.img_w - _input[2][0]
            new_landmarks[1][1] = _input[2][1]
            new_landmarks[2][0] = self.img_w - _input[1][0]
            new_landmarks[2][1] = _input[1][1]
            new_landmarks[3][0] = self.img_w - _input[0][0]
            new_landmarks[3][1] = _input[0][1]
            new_landmarks[4][0] = self.img_w - _input[5][0]
            new_landmarks[4][1] = _input[5][1]
            new_landmarks[5][0] = self.img_w - _input[4][0]
            new_landmarks[5][1] = _input[4][1]
            new_landmarks[6][0] = self.img_w - _input[6][0]
            new_landmarks[6][1] = _input[6][1]
            return new_landmarks
        return _input

    def brighten(self, cropped_img):
        if self.augmentation[2] == '1':
            sign = [-1, 1][random.randrange(2)]
            img = cropped_img * (1 + sign * (random.uniform(0, self.random_brighten_ratio)))
            return img.clip(0, 255)
        return cropped_img

    def normalize(self, _input):
        # Normalize the image.
        if _input.shape == (225, 225, 3):
            img = np.array(_input, dtype=float)
            img = (img / 255) * 2 - 1
            return img

        # Normalize the landmarks.
        landmarks = np.array(_input, dtype=float)
        landmarks = landmarks / self.img_w
        return landmarks

    def denormalize(self, _input):
        # Denormalize the image.
        _input = np.array(_input, dtype=float)
        if _input.shape[0] == 3:
            _input = (_input + 1) / 2 * 255
            return _input.astype(int)

        # Denormalize the landmarks.
        return _input * self.img_w

    def preview(self, idx=-1, is_landmarks_displayed=True):
        if idx == -1:
            idx = random.randint(0, len(self))

        target = self[idx]
        image = self.denormalize(target[0])
        landmarks = target[1]
        landmarks = self.denormalize(landmarks)
        print('Image tensor shape (C, H, W):', image.shape)
        print('Label tensor shape (X, Y):', landmarks.shape)

        channels = image.shape[0]
        h, w = image.shape[1], image.shape[2]

        plt.figure(num='Preview')
        plt.imshow(image.reshape(h, w, channels))

        if is_landmarks_displayed:
            plt.plot(landmarks[0][0], landmarks[0][1], 'ro')
            plt.plot(landmarks[1][0], landmarks[1][1], 'bo')
            plt.plot(landmarks[2][0], landmarks[2][1], 'ro')
            plt.plot(landmarks[3][0], landmarks[3][1], 'bo')
            plt.plot(landmarks[4][0], landmarks[4][1], 'ro')
            plt.plot(landmarks[5][0], landmarks[5][1], 'bo')
            plt.plot(landmarks[6][0], landmarks[6][1], 'go')

        plt.xlim(0, 225)
        plt.ylim(225, 0)
        plt.title('Preview at Index [' + str(idx) + ']')
        plt.draw()
