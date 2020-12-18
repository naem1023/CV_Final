'''
Model3

CNN via pytorch
'''
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

import torch 
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchsummary import summary

class Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.image[idx]
        label = self.labels[idx]
        sample = (image, label)

        return sample
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=10, out_channels=3, kernel_size=3)
        self.pool2 = nn.AdaptiveMaxPool2d(output_size=3)

        self.fc1 = nn.Linear(27, 3)
        self.relu = nn.ReLU6()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool1(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)
        out = self.pool2(out)

        out = torch.flatten(out, 1)

        out = self.fc1(out)

        return out