import matplotlib.pyplot as plt
import cv2
from skimage.feature import greycomatrix, greycoprops
from scipy import signal as sg
import numpy as np


# law's energy
def laws_texture(gray_image):
    # image = cv2.imread('pebbles.jpg')
    # image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (rows, cols) = gray_image.shape[:2]

    # preprocessing image
    # making smoothing filter
    smooth_kernel = (1 / 25) * np.ones((5, 5))

    # smoothing in gray image
    gray_smooth = sg.convolve(gray_image, smooth_kernel, 'same')

    # get difference between gray image and gray smooth
    gray_processed = np.abs(gray_image - gray_smooth)

    # Law's texture fileter
    filter_vectors = np.array([[1, 4, 6, 4, 1],     # L5
                               [-1, -2, 0, 2, 1],   # E5
                               [-1, 0, 2, 0, 1],    # S5
                               [1, -4, 6, -4, 1]])  # R5

    # save 16 filter(5x5)
    filters = list()
    for i in range(4):
        for j in range(4):
            filters.append(np.matmul(filter_vectors[i][:].reshape(5, 1),
                                     filter_vectors[j][:].reshape(1, 5)))

    # convolution and combination of convmap
    conv_maps = np.zeros((rows, cols, 16))
    for i in range(len(filters)):
        conv_maps[:, :, i] = sg.convolve(gray_processed, filters[i], 'same')

    # calculating texture map ( 9+1 )
    texture_maps = list()

    # no ordering eg. L5E5 = E5L5
    texture_maps.append((conv_maps[:, :, 1] + conv_maps[:, :, 4]) // 2)     # L5E5
    texture_maps.append((conv_maps[:, :, 2] + conv_maps[:, :, 8]) // 2)     # L5S5
    texture_maps.append((conv_maps[:, :, 3] + conv_maps[:, :, 12]) // 2)    # L5R5
    texture_maps.append((conv_maps[:, :, 7] + conv_maps[:, :, 13]) // 2)    # E5R5
    texture_maps.append((conv_maps[:, :, 6] + conv_maps[:, :, 9]) // 2)     # E5S5
    texture_maps.append((conv_maps[:, :, 11] + conv_maps[:, :, 14]) // 2)   # S5R5
    texture_maps.append(conv_maps[:, :, 10])                                # S5S5
    texture_maps.append(conv_maps[:, :, 5])                                 # E5E5
    texture_maps.append(conv_maps[:, :, 15])                                # R5R5
    texture_maps.append(conv_maps[:, :, 0])                                 # L5L5

    # calculating Law's texture energy
    TEM = list()
    for i in range(9):
        # get TEM, normalized with L5L5
        # 9 dimension TEM feature
        TEM.append(np.abs(texture_maps[i]).sum() / np.abs(texture_maps[9].sum()))

    return TEM


def glcm_texture():
    # read image
    image = cv2.imread("camera.png", cv2.IMREAD_GRAYSCALE)

    # set patch size
    PATCH_SIZE = 32

    grass_locations = [(370, 454), (373, 22), (444, 244), (455, 455)]

    grass_patches = list()

    for loc in grass_locations:
        grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

    sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]

    sky_patches = list()

    for loc in sky_locations:
        sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                           loc[1]:loc[1] + PATCH_SIZE])

    xs = list()
    ys = list()

    for patch in (grass_patches + sky_patches):
        glcm = greycomatrix(patch, distances=[1], angles=[0], levels=256,
                            symmetric=False, normed=True)
        xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        ys.append(greycoprops(glcm, 'correlation')[0, 0])

    print(xs)
    print(ys)