# exercise 1.c
from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = plt.imread('images/umbrellas.jpg') # 0-255
img = img.astype(np.float64) / 255 # conversion from uint8 to float64

cutoutr=img[130:260, 240:450, 0]
cutoutg=img[130:260, 240:450, 1]
cutoutb=img[130:260, 240:450, 2]

f = plt.figure()
f.add_subplot(2, 3, 1)
plt.imshow(cutoutr, cmap='gray')
plt.title("cutoutr")
f.add_subplot(2, 3, 2)
plt.imshow(cutoutg, cmap='gray')
plt.title("cutoutg")
f.add_subplot(2, 3, 3)
plt.imshow(cutoutb, cmap='gray')
plt.title("cutoutb")
f.add_subplot(2, 3, 5)
plt.imshow(img[130:260, 240:450, :])
plt.title("original")

plt.show()

# Why would you use different color maps?
# Your algorithm does not care about colors, but other features of the image (intensity, edges, shapes,...)
