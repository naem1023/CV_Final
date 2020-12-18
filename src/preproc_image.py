import matplotlib.pyplot as pyplot
import numpy as np
import cv2
import os
from skimage.feature import greycomatrix, greycoprops
from scipy import signal as sg
from glcm_laws_texture import glcm_texture, laws_texture
from tqdm import tqdm

def get_test_image(test_dir, classes, PATCH_SIZE=32, using_glcm=False):
    X_test = []
    Y_test = []

    np.random.seed(1234)

    for idx, texture_name in enumerate(classes):
        test_image_dir = os.path.join(test_dir, texture_name)

        for test_image_name in os.listdir(test_image_dir):
            test_image = cv2.imread(os.path.join(test_image_dir, test_image_name))

            if using_glcm:
                test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                glcm = greycomatrix(test_image_gray, distances=[1], angles=[0], levels=256,
                                    symmetric=False, normed=True)
                X_test.append([greycoprops(glcm, 'dissimilarity')[0, 0],
                                greycoprops(glcm, 'correlation')[0, 0]]
                                + laws_texture(test_image_gray))
                Y_test.append(idx)
            else:
                X_test.append(test_image)
                Y_test.append(idx)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    return X_test, Y_test

def get_train_image(train_dir, classes, PATCH_SIZE=30, using_glcm=False):
    X_train = []
    Y_train = []
    
    np.random.seed(42)

    for idx, texture_name in enumerate(classes):
        train_image_dir = os.path.join(train_dir, texture_name)

        for train_image_name in os.listdir(train_image_dir):
            train_image = cv2.imread(os.path.join(train_image_dir, train_image_name))

            # resize image to 100x100
            image_s = cv2.resize(train_image, (100, 100), interpolation=cv2.INTER_LINEAR)

            for _ in range(10):
                h = np.random.randint(100 - PATCH_SIZE)
                w = np.random.randint(100 - PATCH_SIZE)

                image_p = image_s[h : h + PATCH_SIZE, w : w  + PATCH_SIZE]
                
                if using_glcm:
                    image_p_gray = cv2.cvtColor(image_p, cv2.COLOR_BGR2GRAY)
                    glcm = greycomatrix(image_p_gray, distances=[1], angles=[0], levels=256,
                                    symmetric=False, normed=True)
                    X_train.append([greycoprops(glcm, 'dissimilarity')[0, 0],
                                greycoprops(glcm, 'correlation')[0, 0]]
                                + laws_texture(image_p_gray))
                    Y_train.append(idx)
                else:
                    X_train.append(image_p)
                    Y_train.append(idx)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    return X_train, Y_train

def normalize_image(X_train, Y_train, X_test, Y_test):
    X_train = np.array(X_train) / 128 - 1
    X_train = np.swapaxes(X_train, 1, 3)  # (N, Cin, H, W)
    Y_train = np.array(Y_train)
    print('train data :', X_train.shape)
    print('train label : ', Y_train.shape)

    X_test = np.array(X_train) / 128 - 1
    X_test = np.swapaxes(X_train, 1, 3)  # (N, Cin, H, W)
    Y_test = np.array(Y_train)
    print('test data :', X_test.shape)
    print('test label : ', Y_test.shape)

    return X_train, Y_train, X_test, Y_test


