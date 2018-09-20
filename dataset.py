import torch
import numpy as np
import os
import main
from torch.utils.data import Dataset
from PIL import Image


class AlexNetDataset(Dataset):
    img_w, img_h = 225, 225

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        self.file_path = os.path.join(main.lfw_dataset_path, item['file_path'])
        self.crops = item['crops'].split(' ')
        self.landmarks = item['landmarks'].split(' ')

        img_tensor = np.asarray(Image.open(self.file_path), dtype=np.float32)
        img_tensor = self.crop(img_tensor)
        img_tensor = self.flip(img_tensor)
        img_tensor = self.brighten(img_tensor)
        img_tensor = self.normalize(img_tensor)

        landmarks = np.asarray(self.landmarks)
        landmark_tensors = torch.from_numpy(landmarks).long()

        # 225 x 225 x 3 input image tensor. 7 * 2 landmark tensor.
        return img_tensor, landmark_tensors

    def preview(self, idx=-1, is_landmark_displayed=True):
        if idx == -1:
            # Random select a preview image.
            # idx = random.random within dataset range
            pass

        if is_landmark_displayed:
            # Display landmarks on the preview image.
            pass
        else:
            # Do not display landmarks on the preview image.
            pass

        return self.__getitem__(idx)

    def crop(self, original_image):
        img = original_image

        h, w = img.shape[0], img.shape[1]
        chanel = img.shape[2]

        # Create image tensor
        img_tensor = torch.from_numpy(img)

        # Reshape to (1, 28, 28), the 1 is the channel size
        img_tensor = img_tensor.view((chanel, h, w))
        return img_tensor

    @staticmethod
    def flip(cropped_img):
        return cropped_img

    @staticmethod
    def brighten(cropped_img):
        return cropped_img

    @staticmethod
    def normalize(cropped_img):
        return cropped_img
