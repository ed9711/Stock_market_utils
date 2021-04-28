import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
import glob
import cv2
import torchvision.models as models
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
from pyts.image import RecurrencePlot, GramianAngularField
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from prepData import prep_data
from changeData import *

# data creation
data = pd.read_csv("./data_stocks.csv")
data = data.drop(['DATE'], 1)
data = data.drop(['SP500'], 1)

r, c = data.values.shape
# print(data.columns.values[0])
# print(data.values[:, 0][:-30])
# print(data.values[:, 0][-32])
# print(data.values[:, 0][-31])


if not os.path.exists("./data/original"):
    os.makedirs("./data/original")
if not os.path.exists("./data/change1"):
    os.makedirs("./data/change1")
if not os.path.exists("./data/change2"):
    os.makedirs("./data/change2")
if not os.path.exists("./data/RCplot"):
    os.makedirs("./data/RCplot")

# prediction = []
# for i in range(c): #len(c)
#     # plt.plot(data.values[:, i][:-30])
#     # plt.axis('off')
#     # plt.savefig("./data/original/{}.jpg".format(data.columns.values[i]))
#     # plt.clf()
#     avg_30 = np.sum(data.values[:, i][-30:])/30
#     if data.values[:, i][-31] > avg_30:  # why?
#         prediction.append(0)
#     else:
#         prediction.append(1)

    # rp = RecurrencePlot()
    # X_rp = rp.fit_transform(np.array([list(range(0, r // 5)), data.values[:, i][:r // 5]]))    # large or less data?
    # plt.imshow(X_rp[1], cmap='gist_ncar', origin='lower')
    # plt.tight_layout()
    # plt.axis('off')
    # plt.savefig("./data/RCplot/{}.jpg".format(data.columns.values[i]))
    # plt.clf()

# labels = np.array([data.columns.values, prediction]) # only 20 change to all
# labels = labels.T
# df = pd.DataFrame(labels, columns=["Name", "Prediction"])
# df.to_csv("./labels.csv", index=False)


# image manipulation
# labels = pd.read_csv("./labels.csv")
# labels = labels.values
# image_path = glob.glob(str('./data/original') + str("/*"))
# for i in range(len(image_path)):
#     single_mask = labels[i][1]
#     single_img = image_path[i]
#     img = Image.open(single_img)
#     img1 = cv2.imread(single_img)
#     result = feature_points(np.array(img))
#     result1 = feature_points(img1)
#     keypoints = []
#     for item in result:
#         k = cv2.KeyPoint(item[1],item[0],item[2])
#         keypoints.append(k)
#     keypoints1 = []
#     for item in result1:
#         k = cv2.KeyPoint(item[1], item[0], item[2])
#         keypoints1.append(k)
#     x = cv2.drawKeypoints(img1, np.array(keypoints), None)
#     x1 = cv2.drawKeypoints(img1, np.array(keypoints1), None)
#     location = "./data/change1/{}.jpg".format(labels[i][0])
#     loation1 = "./data/change2/{}.jpg".format(labels[i][0])
#     cv2.imwrite(location, x)
#     cv2.imwrite(loation1, x1)


# data prep
trainLoad, validLoad, testLoad = prep_data("./data/original/", "./labels.csv")
trainLoad1, validLoad1, testLoad1 = prep_data("./data/change1/", "./labels.csv")
trainLoad2, validLoad2, testLoad2 = prep_data("./data/RCplot/", "./labels.csv")
trainLoad3, validLoad3, testLoad3 = prep_data("./data/RCplot1/", "./labels.csv")
# for batch, (images, masks) in enumerate(trainLoad):
    # print(images.cpu().numpy().squeeze().shape)
    # print(images.cpu().numpy().squeeze().transpose(1,2,0)[60])
    # cv2.imshow("x", images.cpu().numpy().squeeze().transpose(1,2,0))
    # cv2.waitKey(0)
    #

net = models.resnet18(pretrained=True)
for param in net.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.SGD(net.fc.parameters(), lr=0.001, momentum=0.9)

net = net.to(device)
start = time.time()

def train_model(model, criterion, optimizer, scheduler, trainLoad, validLoad, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                d = trainLoad
                x = 0
            else:
                model.eval()
                x=1
                d = validLoad # Set model to evaluate mode
            dataset_sizes = [len(trainLoad), len(validLoad)]
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch, (inputs, labels) in enumerate(d):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels[0])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[x]
            epoch_acc = running_corrects.double() / dataset_sizes[x]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model_ft = train_model(net, criterion, optimizer, exp_lr_scheduler,
                       trainLoad, validLoad, num_epochs=25)
model_ft1 = train_model(net, criterion, optimizer, exp_lr_scheduler,
                       trainLoad1, validLoad1, num_epochs=25)
model_ft2 = train_model(net, criterion, optimizer, exp_lr_scheduler,
                       trainLoad2, validLoad2, num_epochs=25)
model_ft3 = train_model(net, criterion, optimizer, exp_lr_scheduler,
                       trainLoad3, validLoad3, num_epochs=25)