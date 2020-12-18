import matplotlib.pyplot as pyplot
import numpy as np
import cv2
import os

train_dir = './texture_data/train'
test_dir = './texutre_data/test'
classes = ['brick', 'grass', 'ground']

PATCH_SIZE = 32

def get_image():
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    np.random.seed(1234)

    for idx, texture_name in enumerate(classes):
        image_dir = os.path.join(train_dir, texutre_name)
        for image_name in os.listdir(image_dir):
            image = cv2.imread(os.path.join(image_dir, image_name))

            X_test.append(image)
            Y_test.append(idX)

            # resize image to 100x100
            image_s = cv2.resize(image, (100, 100), interploation=cv2.INTER_LINEAR)

            for _ in range(10):
                h = np.random.randint(100 - PATCH_SIZE)
                w = np.random.randint(100 - PATCH_SIZE)

                image_p = image_s[h : h + PATCH_SIZE, w : w  + PATCH_SIZE]

                X_train.append(image_p)
                Y_train.append(idx)
    
    return X_train, Y_train, X_test, Y_test

def normalize_image(X_train, Y_train, X_test, Y_test):
    X_train = np.array(X_train) / 128 - 1
    X_train = np.swapaxes(X_train, 1, 3)  # (N, Cin, H, W)
    Y_train = np.array(Y_train)
    print('train data :', X_train.shape)
    print('train label : ', Y_train.shape)

    X_test = np.array(X_train) / 128 - 1
    X_test = np.swapaxes(X_train, 1, 3)  # (N, Cin, H, W)
    Y_test = np.array(Y_train)
    print('test data :', X_testshape)
    print('test label : ', Y_test.shape)

    return X_train, Y_train, X_test, Y_test


