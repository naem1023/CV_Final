'''
Resnet

CNN via pytorch
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import argparse
import time
from copy import deepcopy
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.nn.functional as F

class Dataset(Dataset):
    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        label = self.labels[idx]

        if self.transforms is not None:
            image = self.transforms(image)
        sample = (image, label)

        return sample


class CNN(nn.Module):
    def __init__(self, input_dim, dim1, dim2, output_dim, stride=2, kernel_size=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=dim1, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(dim1)

        self.conv2 = nn.Conv2d(in_channels=dim1, out_channels=dim1, kernel_size=kernel_size)
        self.pool1 = nn.MaxPool2d(kernel_size=kernel_size-1, stride=stride)

        self.conv3 = nn.Conv2d(in_channels=dim1, out_channels=dim1, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(dim1)

        self.conv4 = nn.Conv2d(in_channels=dim1, out_channels=input_dim, kernel_size=kernel_size)
        self.pool2 = nn.AdaptiveMaxPool2d(output_size=input_dim)

        self.bn3 = nn.BatchNorm2d(input_dim)

        self.fc1 = nn.Linear(dim2, input_dim)
        self.relu = nn.ReLU6()

    def forward(self, x):
        # x = torch.flatten(x, 1)
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.pool1(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.pool2(out)

        shortcut = torch.flatten(shortcut, 1)
        out += shortcut

        out = torch.flatten(out, 1)

        out = self.fc1(out)

        return out