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

def make_model(X_train, Y_train, X_test, Y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 10
    learning_rate = 0.001
    n_epoch = 100

    train_data = Dataset(images=X_train, labels=Y_train)
    test_data = Dataset(images=X_test, labels=Y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    net = CNN()
    net.to(device)
    summary(net, (3, 32, 32), device='cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    creiterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    test_looses = []
    test_accs = []

    for epoch in range(n_epoch):
        tarin_loss = 0.0
        evaluation = []

        for idx, data in enumerate(train_loader, 0):
            features, labels = data
            labels = labels.long().to(device)
            features = features.to(device)
            optimizer.zero_grad()

            outputs = net(features.to(torch.float))

            _, predicted = torch.max(outputs.cpu().data, 1)

            evaluation.append((predicted==labels.cpu()).tolist())
            loss = creiterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss = train_loss / (idx+1)
        evaluation = [item for sublist in evaluation for item in sublist]
        train_acc = sum(evaluation) / len(evaluation)

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        if (epoch + 1) % 1 == 0:
            test_loss = 0.0
            evalutaion = []

            for idx, data in enumerate(test_loader, 0):
                features, labels = data
                labels = labels.long().to(device)

                features = features.to(device)

                outputs = net(features.to(torch.float))

                _, predicted = torch.max(outputs.cpu().data, 1)

                evaluation.append((predicted==labels.cpu()).tolist())
                loss = creiterion(outputs, labels)

                test_loss += loss.item()
            
            test_loss = test_loss / (idx+1)
            evaluation = [item for sublist in evaluation for item in sublist]
            test_acc = sum(evaluation) / len(evaluation)

            test_losses.append(test_loss)
            test_accs.append(test_acc)

            print('[%d %3d]\tloss: %.4\tAccuracy : %.4f\t\tval-loss: %.4f\tval-Accuracy : %.4f' %
                    (epoch + 1, n_epoch, train_loss, train_acc, test_loss, test_acc))


def make_graph(train_losses, test_losses, train_accs, test_accs):
    plt.plot(range(len(train_losses)), train_losses, label='train loss')
    plt.plot(range(len(test_losses)), test_losses, label='test loss')
    plt.legend()
    plt.show()
    plt.savefig('model3_loss_graph.png')

    plt.plot(range(len(train_accs)), train_accs, label='train acc')
    plt.plot(range(len(test_accs)), test_acc, label='test acc')
    plt.legend()
    plt.show()
    plt.savefig('model3_acc_graph.png')

    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18CA6-tLrtRSqB-wwQRZnKlnfrX9CtB2Y' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18CA6-tLrtRSqB-wwQRZnKlnfrX9CtB2Y" -O dataset.zip && rm -rf /tmp/cookies.txt

    https://drive.google.com/file/d/18CA6-tLrtRSqB-wwQRZnKlnfrX9CtB2Y/view?usp=sharing