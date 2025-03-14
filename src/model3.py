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
        image = self.images[idx]
        label = self.labels[idx]
        sample = (image, label)

        return sample
    
class CNN(nn.Module):
    def __init__(self, input_dim, dim1, dim2, output_dim, stride=2, kernel_size=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=dim1, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(in_channels=dim1, out_channels=dim1, kernel_size=kernel_size)
        self.pool1 = nn.MaxPool2d(kernel_size=kernel_size-1, stride=stride)

        self.conv3 = nn.Conv2d(in_channels=dim1, out_channels=dim1, kernel_size=kernel_size)
        self.conv4 = nn.Conv2d(in_channels=dim1, out_channels=input_dim, kernel_size=kernel_size)
        self.pool2 = nn.AdaptiveMaxPool2d(output_size=input_dim)

        self.fc1 = nn.Linear(dim2, output_dim)
        self.relu = nn.ReLU6()

    def forward(self, x):
        # x = torch.flatten(x, 1)
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