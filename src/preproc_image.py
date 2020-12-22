import matplotlib.pyplot as pyplot
import numpy as np
import cv2
import os
from skimage.feature import greycomatrix, greycoprops
from scipy import signal as sg
from src.glcm_laws_texture import glcm_texture, laws_texture
from tqdm import tqdm

def get_test_image(test_dir, classes, using_glcm=False):
    X_test = []
    Y_test = []

    for idx, texture_name in tqdm(enumerate(classes)):
        test_image_dir = os.path.join(test_dir, texture_name)

        for test_image_name in os.listdir(test_image_dir):
            test_image = cv2.imread(os.path.join(test_image_dir, test_image_name))
            test_image = cv2.resize(test_image, (150, 150), interpolation=cv2.INTER_LINEAR)
            if using_glcm:
                test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                glcm = greycomatrix(test_image_gray, distances=[1], angles=[0], levels=256,
                                    symmetric=False, normed=True)
                X_test.append([greycoprops(glcm, 'dissimilarity')[0, 0],
                                    greycoprops(glcm, 'correlation')[0, 0]
                                    , greycoprops(glcm, 'homogeneity')[0, 0]
                                    , greycoprops(glcm, 'contrast')[0, 0]]
                                    + laws_texture(test_image_gray))
                Y_test.append(idx)
            else:
                test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                X_test.append(test_image)
                Y_test.append(idx)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    print('test data :', X_test.shape)
    print('test label : ', Y_test.shape)
    return X_test, Y_test

def get_train_image(train_dir, classes, PATCH_SIZE=30, using_glcm=False):
    X_train = list()
    Y_train = list()
    
    np.random.seed(42)

    for idx, texture_name in enumerate(classes):
        train_image_dir = os.path.join(train_dir, texture_name)
        print(train_image_dir)

        for train_image_name in tqdm(os.listdir(train_image_dir)):
            train_image = cv2.imread(os.path.join(train_image_dir, train_image_name))
            # print(np.array(train_image).shape)

            if PATCH_SIZE is not None:
                # resize image to 100x100
                image_s = cv2.resize(train_image, (150, 150), interpolation=cv2.INTER_LINEAR)

                for _ in range(12):
                    h = np.random.randint(150 - PATCH_SIZE)
                    w = np.random.randint(150 - PATCH_SIZE)

                    image_p = image_s[h : h + PATCH_SIZE, w : w  + PATCH_SIZE]
                    
                    if using_glcm:
                        image_p_gray = cv2.cvtColor(image_p, cv2.COLOR_BGR2GRAY)
                        glcm = greycomatrix(image_p_gray, distances=[1], angles=[0], levels=256,
                                        symmetric=False, normed=True)
                        X_train.append([greycoprops(glcm, 'dissimilarity')[0, 0],
                                    greycoprops(glcm, 'correlation')[0, 0]
                                    , greycoprops(glcm, 'homogeneity')[0, 0]
                                    , greycoprops(glcm, 'contrast')[0, 0]]
                                    + laws_texture(image_p_gray))
                        Y_train.append(idx)
                    else:
                        image_p = cv2.cvtColor(image_p, cv2.COLOR_BGR2RGB)
                        X_train.append(image_p)
                        Y_train.append(idx)
            else:
                train_image_gray = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
                train_image = cv2.resize(train_image, (150, 150), interpolation=cv2.INTER_LINEAR)
                # print(np.array(train_image).shape)
                X_train.append(train_image)
                Y_train.append(idx)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # X_train = np.vstack(X_train)
    # Y_train = np.vstack(Y_train)

    print('train data :', X_train.shape)
    # print(X_train.shape)
    # print(X_train)
    print('train label : ', Y_train.shape)
    # print(Y_train.shape)
    

    return X_train, Y_train

def normalize_image(X_train, Y_train, X_test, Y_test):
    X_train = X_train / 128 - 1
    # X_train = np.swapaxes(X_train, 1, 3)  # (N, Cin, H, W)
    print('train data :', X_train.shape)
    print('train label : ', Y_train.shape)

    X_test = X_test / 128 - 1
    # X_test = np.swapaxes(X_test, 1, 3)  # (N, Cin, H, W)
    print('test data :', X_test.shape)
    print('test label : ', Y_test.shape)

    return X_train, Y_train, X_test, Y_test