import numpy as np
import torch
import torch.nn as nn  # neural network lib.
import torch.nn.functional as F  # common functions (e.g. relu, drop-out, softmax...)
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import matplotlib.pyplot as plt
from random import randint

print("hello")

# Set default tenosr type, 'torch.cuda.FloatTensor' is the GPU based FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# make a data set . extract file paths and landmark feature cordonates from LFW train annotations

lfw_dataset_path = '/home/vramiyas/PycharmProjects/Project 1/lfw'
test_landmark_path = os.path.join(lfw_dataset_path, 'LFW_annotation_test.txt')
train_landmark_path = os.path.join(lfw_dataset_path, 'LFW_annotation_train.txt')
training_ratio = 0.8
aug_types = ['001', '010', '100', '011', '101', '110', '111']  # nnn - random crop , flip,brightness

training_validation_data_list = []

testing_data_list = []  # Read training and validation data.
with open(train_landmark_path, "r") as file:
    for line in file:
        tokens = line.split('\t')
        if len(tokens) == 3:
            file_path = tokens[0]
            crops = tokens[1].split()
            landmarks = tokens[2].split()
            training_validation_data_list.append(
                {'file_path': file_path, 'crops': crops, 'landmarks': landmarks, 'aug_types': '000'})
            random.shuffle(aug_types)
            max_augs = randint(3, 7)
            itr = 0
            for idx in aug_types:
                training_validation_data_list.append(
                    {'file_path': file_path, 'crops': crops, 'landmarks': landmarks, 'aug_types': idx})
                itr = itr + 1
                if itr == max_augs:
                    break


    def crop( image, crops, landmarks, offset=False):
        # Case that a image need to be cropped.
        crops_offset = crops
        print(crops)
        print("typejhihn1j32h312houi")
        print(type(image))
        if offset == True:
            for index in range(0,3):
                rand_offset = randint(-5, 5)
                crops_offset[index] = crops[index] + rand_offset
        else:
            crops_offset = crops

        print(crops_offset)

        img = image
        img = img.crop((crops_offset))
        landmarks_offset = [crops_offset[0], crops_offset[1]]
        landmarks_offset = np.tile(landmarks_offset, 7)
        cropped_landmarks = landmarks - landmarks_offset
        w,h = img.size
        print("h,w")
        print(w,h)
        img = img.resize((225, 225), Image.ANTIALIAS)
        ratio_width = w/225
        ratio_height = h/225
        landmark_offset_ratio = [ratio_width,ratio_height]
        landmark_offset_ratio = np.tile(landmark_offset_ratio,7)
        cropped_landmarks = cropped_landmarks / landmark_offset_ratio
        print(ratio_height,ratio_width)
        return img, cropped_landmarks

    random.shuffle(training_validation_data_list)
    item = training_validation_data_list[0]
    print( training_validation_data_list[0])
    dir_name_trim_length = -9
    file_name = item['file_path']
    file_path = os.path.join(lfw_dataset_path, file_name[:dir_name_trim_length], file_name)
    crops = np.asarray(item['crops'])
    crops = crops.astype(int)
    print("fsafsadf")
    print(crops)
    landmarks = np.asarray(item['landmarks'])
    landmarks = landmarks.astype(float)
    aug_types = item['aug_types']

    aug_type_list = list(aug_types)

    # Prepare image tensor.
    orig_img = Image.open(file_path)
    print("dafnjkanafdasfdafsa")
    print(type(orig_img))
    crop_landmarks =[]


    if aug_type_list[0] == '1':
        cropped_img, crop_landmarks = crop(orig_img, crops, landmarks, True)
    else:
        cropped_img, crop_landmarks = crop(orig_img, crops, landmarks, False)

    plt.imshow(orig_img)
    plt.scatter(x=[landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8], landmarks[10], landmarks[12]],
                y=[landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9], landmarks[11], landmarks[13]],
                c='r', s=10)
    plt.show()

    plt.imshow(cropped_img)
    plt.scatter(x=[crop_landmarks[0], crop_landmarks[2], crop_landmarks[4], crop_landmarks[6], crop_landmarks[8],
                   crop_landmarks[10], crop_landmarks[12]],
                y=[crop_landmarks[1], crop_landmarks[3], crop_landmarks[5], crop_landmarks[7], crop_landmarks[9],
                   crop_landmarks[11], crop_landmarks[13]],
                c='r', s=10)

    plt.show()
