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

import model1

def make_model(X_train, Y_train, X_test, Y_test, _epoch, model_num):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 10
    learning_rate = 0.001
    n_epoch = _epoch

    if model_num == 1:
        train_data = model1.TextureDataset(features=X_train, labels=Y_train)
        test_data = model1.TextureDataset(features=X_test, labels=Y_test)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        net = model1.MLP(11, 8, 3)
        net.to(device)
        summary(net, (11,), device='cuda' if torch.cuda.is_available() else 'cpu')    

        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
        creiterion = nn.CrossEntropyLoss()
    elif model_num == 2:
        pass
    elif model_num == 3:
        net = CNN()
        net.to(device)
        summary(net, (3, 32, 32), device='cuda' if torch.cuda.is_available() else 'cpu')

        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        creiterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

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
