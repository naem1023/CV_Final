from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
from scipy import signal as sg
import itertools
import numpy as np
import cv2
import os
from glcm_texutre import laws_texture

train_dir = './texture_data/train'
test_dir = './texture_data/test'
classes = ['brick', 'grass', 'ground', 'water', 'wood']

PATCH_SIZE = 16

np.random.seed(1234)


def get_laws_texture():
    X_train = []
    Y_train = []

    for idx, texture_name in enumerate(classes):
        image_dir = os.path.join(train_dir, texture_name)

        # get all image in class dir
        for image_name in os.listdir(image_dir):
            image = cv2.imread(os.path.join(image_dir, image_name))

            # resize image to 100x100
            image_s = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)

            # crop and get TEM 20 times
            for _ in range(20):
                # get random coordinate point
                h = np.random.randint(100 - PATCH_SIZE)
                w = np.random.randint(100 - PATCH_SIZE)

                # slice image using random patch coordinate point
                image_p = image_s[h:h + PATCH_SIZE, w:w + PATCH_SIZE]

                # convert image to gray
                image_p_gray = cv2.cvtColor(image_p, cv2.COLOR_BGR2GRAY)
                # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

                # get laws energy
                X_train.append(laws_texture(image_p_gray))
                Y_train.append(idx)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    print('train data : ', X_train.shape)

    return X_train, Y_train

def get_glcm_texture():
    X_train = []
    Y_train = []

    for idx, texture_name in enumerate(classes):
        image_dir = os.path.join(train_dir, texture_name)

        # get all image in class dir
        for image_name in os.listdir(image_dir):
            image = cv2.imread(os.path.join(image_dir, image_name))

            # resize image to 100x100
            image_s = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)

            # crop and get glcm 20 times
            for _ in range(20):
                # get random coordinate point
                h = np.random.randint(100 - PATCH_SIZE)
                w = np.random.randint(100 - PATCH_SIZE)

                # slice image using random patch coordinate point
                image_p = image_s[h:h + PATCH_SIZE, w:w + PATCH_SIZE]

                # convert image to gray
                image_p_gray = cv2.cvtColor(image_p, cv2.COLOR_BGR2GRAY)
                # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

                # get glcm
                glcm = greycomatrix(image_p_gray, distances=[1], angles=[0], levels=256,
                                    symmetric=False, normed=True)
                X_train.append([greycoprops(glcm, 'dissimilarity')[0, 0],
                                greycoprops(glcm, 'correlation')[0, 0]])
                Y_train.append(idx)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    print('train data : ', X_train.shape)

    return X_train, Y_train

def make_train_data():
    X_train = []
    Y_train = []

    for idx, texture_name in enumerate(classes):
        image_dir = os.path.join(train_dir, texture_name)
        for image_name in os.listdir(image_dir):
            image = cv2.imread(os.path.join(image_dir, image_name))
            image_s = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)

            for _ in range(10):
                h = np.random.randint(100 - PATCH_SIZE)
                w = np.random.randint(100 - PATCH_SIZE)

                image_p = image_s[h:h + PATCH_SIZE, w:w + PATCH_SIZE]
                image_p_gray = cv2.cvtColor(image_p, cv2.COLOR_BGR2GRAY)
                # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
                glcm = greycomatrix(image_p_gray, distances=[1], angles=[0], levels=256,
                                    symmetric=False, normed=True)
                X_train.append([greycoprops(glcm, 'dissimilarity')[0, 0],
                                greycoprops(glcm, 'correlation')[0, 0]]
                               + laws_texture(image_p_gray))
                Y_train.append(idx)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    print('train data : ', X_train.shape)
    print('train label : ', Y_train.shape)

    return X_train, Y_train


def make_test_data():
    X_test = []
    Y_test = []

    for idx, texture_name in enumerate(classes):
        image_dir = os.path.join(train_dir, texture_name)
        for image_name in os.listdir(image_dir):
            image = cv2.imread(os.path.join(image_dir, image_name))
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            glcm = greycomatrix(image_gray, distances=[1], angles=[0], levels=256,
                                symmetric=False, normed=True)
            X_test.append([greycoprops(glcm, 'dissimilarity')[0, 0],
                           greycoprops(glcm, 'correlation')[0, 0]]
                          + laws_texture(image_gray))
            Y_test.append(idx)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    print('test data : ', X_test.shape)
    print('test label : ', Y_test.shape)

    return X_test, Y_test
