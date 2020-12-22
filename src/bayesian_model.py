from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
from scipy import signal as signalimport
import itertools
import numpy as np
import cv2
import os
from tqdm import tqdm

def get_bayesian_value(X_train, Y_train, classes):
    priors = []
    covariances = []
    means = []

    for i in range(len(classes)):
        X = X_train[Y_train == i]
        priors.append((len(X) / len(X_train)))
        means.append(np.mean(X, axis=0))
        covariances.append(np.cov(np.transpose(X), bias=True))

    return priors, covariances, means


def likelihood(x, prior, mean, cov):
    return -0.5 * np.linalg.multi_dot([np.transpose(x - mean), 
        np.linalg.inv(cov), (x - mean)]) - 0.5 * np.log(np.linalg.det(cov)) + np.log(prior)

def plot_confusion_matrix(cm, target_names=None, labels=True):
    accuracy = np.trace(cm) / float(np.sum(cm))

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6,4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    thresh = cm.max() / 2

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "{:,}".format(cm[i,j]), ha="center",color="white" if cm[i,j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig('bayesian_graph.png')
        
def bayesian(X_train, Y_train, X_test, Y_test, classes):
    Y_pred = []

    priors, covariances, means = get_bayesian_value(X_train, Y_train, classes)
    for i in tqdm(range(len(X_test))):
        likelihoods = []
        for j in range(len(classes)):
            likelihoods.append(likelihood(X_test[i], priors[j], means[j], covariances[j]))
        Y_pred.append(likelihoods.index(max(likelihoods)))

    Y_pred = np.array(Y_pred)
    print("Y_pred=",Y_pred.shape)
    acc = accuracy_score(Y_test, Y_pred)
    print("Bayesian Accuracy=", acc)

    with open("bayesian_output.txt", 'w') as file:
        file.write(str(Y_pred) + '\n')
        file.write(str(acc) + "\n")
    
    plot_confusion_matrix(confusion_matrix(Y_test, Y_pred), target_names=classes)