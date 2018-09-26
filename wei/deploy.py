import lfwnet
import main
import dataset
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import random

n_testing_data = 128
n_detection_range = 50

# Load trained model.
test_net = lfwnet.LfwNet()
test_net_state = torch.load('lfw_net.pth')
test_net.load_state_dict(test_net_state)

# Read testing data.
oracle_data_list = []
with open(main.test_landmark_path, "r") as file:
    for line in file:
        tokens = line.split('\t')
        if len(tokens) == 3:
            file_path = tokens[0]
            crops = ' '.join(tokens[1].split())
            landmarks = ' '.join(tokens[2].split())
            oracle_data_list.append(
                {'file_path': file_path, 'crops': crops, 'landmarks': landmarks, 'augmentation': '000'})
prediction_data_list = oracle_data_list

# Generate predictions.
test_dataset = dataset.AlexNetDataset(oracle_data_list)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=n_testing_data, shuffle=False, num_workers=6)
loss_raw = []
for test_batch_idx, (test_input, test_oracle) in enumerate(test_data_loader):
    test_net.eval()
    test_net.cuda()
    predictions = test_net.forward(test_input.cuda())

    # Random preview.
    if test_batch_idx == 0:
        idx = random.randint(0, n_testing_data)
        predictions[idx] *= dataset.img_w
        landmarks = np.array(predictions[idx].detach())

        test_dataset.preview(idx, is_landmarks_displayed=False)
        marker_size = 10
        r_dot, = plt.plot(landmarks[0][0], landmarks[0][1], 'ro', markersize=marker_size)
        b_dot, = plt.plot(landmarks[1][0], landmarks[1][1], 'bo', markersize=marker_size)
        plt.plot(landmarks[2][0], landmarks[2][1], 'ro', markersize=marker_size)
        plt.plot(landmarks[3][0], landmarks[3][1], 'bo', markersize=marker_size)
        plt.plot(landmarks[4][0], landmarks[4][1], 'ro', markersize=marker_size)
        plt.plot(landmarks[5][0], landmarks[5][1], 'bo', markersize=marker_size)
        g_dot, = plt.plot(landmarks[6][0], landmarks[6][1], 'go', markersize=marker_size)

        plt.legend([r_dot, b_dot, g_dot],
                   ['Left Dot', 'Right Dot', 'Nose Dot'],
                   bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()

    # Compute overall loss.
    batch_loss = np.linalg.norm(np.array(predictions.detach()) - np.array(test_oracle.detach()), axis=2)
    batch_loss = batch_loss.flatten()
    batch_loss *= 225
    loss_raw.extend(batch_loss.tolist())

# Compute loss plot.
loss_raw = np.array(loss_raw).flatten()
loss_plot = []
for step in range(1, n_detection_range):
    loss_plot.append((step, len(np.where(loss_raw < step)[0]) / len(loss_raw)))
loss_plot = np.asarray(loss_plot)
plt.figure(num='Percentage of Detected Key-points')
plt.title('Percentage of Detected Key-points')
plt.xlabel('L2 Distance From Detected Points to Ground Truth Points')
plt.ylabel('Percentage')
axes = plt.gca()
axes.set_xticks(np.arange(0, n_detection_range, 5))
axes.set_yticks(np.arange(-0.1, 1.1, 0.1))
axes.set_xlim([0, n_detection_range])
axes.set_ylim([-0.05, 1.05])
plt.grid()
plt.plot(loss_plot[:, 0], loss_plot[:, 1])

plt.show()
