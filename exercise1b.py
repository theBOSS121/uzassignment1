# exercise 1.b
from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    v = r * 0.2989 + g * 0.5870 + b * 0.1140
    rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = v, v, v
    return rgb

# def rgb2grayAvg(rgb):
#     return np.dot(rgb[...,:3], [0.33, 0.33, 0.33])

img = plt.imread('images/umbrellas.jpg') # 0-255
# img = cv2.imread('images/umbrellas.jpg') # bgr format 0-255
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # bgr => rgb
img = img.astype(np.float64) / 255 # conversion from uint8 to float64
grey = rgb2gray(np.copy(img))
# grey2 = rgb2grayAvg(img)

f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(img)
plt.title("RGB")

# plt.set_cmap('gray')
f.add_subplot(1, 2, 2)
plt.imshow(grey)
plt.title("Grey")

# f.add_subplot(2, 2, 3)
# plt.imshow(grey2)
# plt.title("Grey2")
plt.show()