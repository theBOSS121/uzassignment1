# exercise 2.a
from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return r * 0.2989 + g * 0.5870 + b * 0.1140

img = plt.imread('images/bird.jpg') # 0-255
grey = rgb2gray(img)
threshold = 52
# grey[grey < threshold] = 0
# grey[grey >= threshold] = 1
grey = np.where(grey < threshold, 0, 1) # can be used alternatively

f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(img)
plt.title("RGB")

f.add_subplot(1, 2, 2)
plt.imshow(grey, cmap="gray")
plt.title("Binary mask")

plt.show()