# exercise 3.a and 3.b
from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt

def rgb2gray(rgb):
    rgb = rgb.astype(np.float64)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    v = (r + g + b) / 3
    return v.astype(np.uint8)
# def rgb2gray(rgb):
#     r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
#     return  r * 0.2989 + g * 0.5870 + b * 0.1140


img = plt.imread('images/bird.jpg') # 0-255
grey = rgb2gray(np.copy(img))

threshold = 52
mask = np.where(grey < threshold, 0, 1) # can be used alternatively
mask = mask.astype(np.uint8)

n = 15
# se = np.ones((n,n), np.uint8) # create a square structuring element
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))

img_eroded = cv2.erode(mask, se)
img_dilated = cv2.dilate(mask, se)

img_eroded_dilated = cv2.dilate(img_eroded, se)
img_dilated_eroded = cv2.erode(img_dilated, se)

f = plt.figure()
f.add_subplot(2, 3, 1)
plt.imshow(grey, cmap="gray")
plt.title("Grey")

f.add_subplot(2, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")

f.add_subplot(2, 3, 3)
plt.imshow(img_eroded, cmap="gray")
plt.title("Eroded")

f.add_subplot(2, 3, 4)
plt.imshow(img_dilated, cmap="gray")
plt.title("Dilated")

f.add_subplot(2, 3, 5)
plt.imshow(img_eroded_dilated, cmap="gray")
plt.title("Eroded-dilated")

f.add_subplot(2, 3, 6)
plt.imshow(img_dilated_eroded, cmap="gray")
plt.title("Dilated-eroded")

plt.show()

# Question: Based on the results, which order of erosion and dilation operations produces opening and which closing?
# erosion => dilation = opening
# dilation => erosion = closing