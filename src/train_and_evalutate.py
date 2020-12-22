'''
test model
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

import src.model1
import src.model2
import src.model3
import src.resnet

from torchvision import transforms
from torchvision.datasets import CIFAR10

def make_model(X_train, Y_train, X_test, Y_test, _epoch, model_num, PATCH_SIZE=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 10
    learning_rate = 0.001
    n_epoch = _epoch

    if model_num == 1:
        train_data = src.model1.TextureDataset(features=X_train, labels=Y_train)
        test_data = src.model1.TextureDataset(features=X_test, labels=Y_test)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        net = src.model1.MLP(150*150*1, 8, 4)
        net.to(device)
        summary(net, (150,150,1), device='cuda' if torch.cuda.is_available() else 'cpu')    

        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
        creiterion = nn.CrossEntropyLoss()
    elif model_num == 2:
        train_data = src.model2.Dataset(images=X_train, labels=Y_train)
        test_data = src.model2.Dataset(images=X_test, labels=Y_test)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        net = src.model2.MLP(150*150*3, 64, 128, 4)
        net.to(device)
        summary(net, (150, 150, 3), device='cuda' if torch.cuda.is_available() else 'cpu')    

        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        creiterion = nn.CrossEntropyLoss()
    elif model_num == 3:
        X_train = np.swapaxes(X_train, 1, 3)  # (N, Cin, H, W)
        X_test = np.swapaxes(X_test, 1, 3)  # (N, Cin, H, W)
        print("before train X_train", X_train.shape)
        print("before train X_test", X_test.shape)

        train_data = src.model3.Dataset(images=X_train, labels=Y_train)
        test_data = src.model3.Dataset(images=X_test, labels=Y_test)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        net = src.model3.CNN(3, 10, 27, 4)
        net.to(device)
        summary(net, (3, 150, 150), device='cuda' if torch.cuda.is_available() else 'cpu')

        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        creiterion = nn.CrossEntropyLoss()
    elif model_num == 4:
        X_train = np.swapaxes(X_train, 1, 3)  # (N, Cin, H, W)
        X_test = np.swapaxes(X_test, 1, 3)  # (N, Cin, H, W)

        print("before train X_train", X_train.shape)
        print("before train X_test", X_test.shape)

        train_data = src.resnet.Dataset(images=X_train, labels=Y_train)
        test_data = src.resnet.Dataset(images=X_test, labels=Y_test)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        net = src.resnet.CNN(3, 10, 27, 4)
        net.to(device)

        summary(net, (3,150,150), device='cuda' if torch.cuda.is_available() else 'cpu')

        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        creiterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    with open('model' + str(model_num) + '_output.txt', 'w') as output_file:
        for epoch in range(n_epoch):
            train_loss = 0.0
            evaluation = []
            net.train()

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

            # test
            if (epoch + 1) % 1 == 0:
                test_loss = 0.0
                evaluation = []

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

                print('[%d %3d]\tloss: %.4f\tAccuracy : %.4f\t\tval-loss: %.4f\tval-Accuracy : %.4f' % 
                (epoch + 1, n_epoch, train_loss, train_acc, test_loss, test_acc))

                output_file.write('[%d %3d]\tloss: %.4f\tAccuracy : %.4f\t\tval-loss: %.4f\tval-Accuracy : %.4f\n' % 
                (epoch + 1, n_epoch, train_loss, train_acc, test_loss, test_acc))

    return train_losses, test_losses, train_accs, test_accs

def make_graph(train_losses, test_losses, train_accs, test_accs, model_num):
    plt.plot(range(len(train_losses)), train_losses, label='train loss')
    plt.plot(range(len(test_losses)), test_losses, label='test loss')
    plt.legend()
    plt.show()
    plt.savefig('model' + str(model_num) + '_loss_graph.png')

    # erase all
    plt.clf()

    plt.plot(range(len(train_accs)), train_accs, label='train acc')
    plt.plot(range(len(test_accs)), test_accs, label='test acc')
    plt.legend()
    plt.show()
    plt.savefig('model' + str(model_num) + '_acc_graph.png')
