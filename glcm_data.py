import matplotlib.pyplot as plt
import cv2
from skimage.feature import greycomatrix, greycoprops
from scipy import signal as sg
import numpy as np
from test import make_train_data, make_test_data, get_glcm_texture, get_laws_texture
from texture import laws_texture
import csv

from texture import laws_texture, glcm_texture

def glcm():
    # 'brick', 'grass', 'ground', 'water', 'wood'
    #  class number is 0, 1, 2, 3, 4

    x_glcm, y_glcm = get_glcm_texture()
    x_laws, y_laws = get_laws_texture()

    # print(x_glcm, y_glcm)
    # print(x_laws, y_laws)
    x_glcm_file = open('X_glcm_texture.csv', 'w', encoding='utf8', newline='')
    y_glcm_file = open('Y_glcm_texture.csv', 'w', encoding='utf8', newline='')

    x_laws_file = open('X_laws_texture.csv', 'w', encoding='utf8', newline='')
    y_laws_file = open('Y_laws_texture.csv', 'w', encoding='utf8', newline='')

    x_glcm_wr = csv.writer(x_glcm_file)
    y_glcm_wr = csv.writer(y_glcm_file)

    x_laws_wr = csv.writer(x_laws_file)
    y_laws_wr = csv.writer(y_laws_file)

    for row in x_glcm:
        x_glcm_wr.writerow(row)
    for row in y_glcm:
        y_glcm_wr.writerow([row])

    for row in x_laws:
        x_laws_wr.writerow(row)
    for row in y_laws:
        y_laws_wr.writerow([row])

    x_glcm_file.close()
    y_glcm_file.close()
    x_laws_file.close()
    y_laws_file.close()

    # X_train, Y_train = make_train_data()
    # X_test, Y_test = make_test_data()
    #
    # print(X_train, Y_train)
    # print(X_test, Y_test)
    #
    # x_train_file = open('X_train_output.csv', 'w', encoding='utf-8', newline='')
    # y_train_file = open('Y_train_output.csv', 'w', encoding='utf-8', newline='')
    # x_test_file = open('x_test_output.csv', 'w', encoding='utf-8', newline='')
    # y_test_file = open('y_test_output.csv', 'w', encoding='utf-8', newline='')
    #
    # x_train_wr = csv.writer(x_train_file)
    # y_train_wr = csv.writer(y_train_file)
    #
    # x_test_wr = csv.writer(x_test_file)
    # y_test_wr = csv.writer(y_test_file)
    #
    # for row in X_train:
    #     # print(type(row))
    #     x_train_wr.writerow(row)
    #
    # # for row in Y_train:
    # #     print(type(row))
    # #     y_train_wr.writerow(row)
    #
    # for row in X_test:
    #     x_test_wr.writerow(row)
    #
    # # for row in Y_test:
    # #     y_test_wr.writerow(row)
    #
    # x_train_file.close()
    # y_train_file.close()
    # x_test_file.close()
    # y_test_file.close()

