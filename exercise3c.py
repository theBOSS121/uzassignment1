# exercise 3.c
from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt

def rgb2gray(rgb):
    rgb = rgb.astype(np.float64)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    v = (r + g + b) / 3
    return v.astype(np.uint8)

def immask(img, mask):
    m = np.expand_dims(mask, axis=2)
    return np.where(m == 1, img, [0, 0, 0])


img = plt.imread('images/bird.jpg') # 0-255
grey = rgb2gray(np.copy(img))

threshold = 52
mask = np.where(grey < threshold, 0, 1) # can be used alternatively
mask = mask.astype(np.uint8)

n = 15
# se = np.ones((n,n), np.uint8) # create a square structuring element
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))

# img_eroded = cv2.erode(mask, se)
# img_eroded_dilated = cv2.dilate(img_eroded, se)
img_dilated = cv2.dilate(mask, se)
img_dilated_eroded = cv2.erode(img_dilated, se)
imgMasked = immask(img, img_dilated_eroded)


f = plt.figure()
f.add_subplot(1, 1, 1)
plt.imshow(imgMasked)
plt.show()
