import os
import random
import torch.utils.data
import dataset
import matplotlib.pyplot as plt
from random import randint

lfw_dataset_path = 'lfw'
test_landmark_path = os.path.join(lfw_dataset_path, 'LFW_annotation_test.txt')
train_landmark_path = os.path.join(lfw_dataset_path, 'LFW_annotation_train.txt')
training_ratio = 0.8
aug_types = ['001', '010', '100', '011', '101', '110', '111']  # nnn - random crop , flip,brightness

if __name__ == '__main__':
    training_validation_data_list = []
    testing_data_list = []

    # Read training and validation data.
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
                        exit()

    # Read testing data.
    with open(test_landmark_path, "r") as file:
        for line in file:
            tokens = line.split('\t')
            if len(tokens) == 3:
                file_path = tokens[0]
                crops = tokens[1]
                landmarks = ' '.join(tokens[2].split())
                testing_data_list.append({'file_path': file_path, 'crops': crops, 'landmarks': landmarks})

    random.shuffle(training_validation_data_list)
    random.shuffle(testing_data_list)
    total_training_validation_items = len(training_validation_data_list)

    # Training dataset
    n_train_sets = training_ratio * total_training_validation_items
    train_set_list = training_validation_data_list[: int(n_train_sets)]

    # Validation dataset
    n_valid_sets = (1 - training_ratio) * total_training_validation_items
    valid_set_list = training_validation_data_list[int(n_train_sets): int(n_train_sets + n_valid_sets)]

    # Testing dataset
    test_set_list = testing_data_list

    train_dataset = dataset.AlexNetDataset(train_set_list)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=6)
    print('Total training items', len(train_dataset), ', Total training batches per epoch:', len(train_data_loader))

    valid_set = dataset.AlexNetDataset(valid_set_list)
    valid_data_loader = torch.utils.data.DataLoader(valid_set, batch_size=32, shuffle=True, num_workers=6)
    print('Total validation set:', len(valid_set))

    train_dataset.preview()

    plt.show()
