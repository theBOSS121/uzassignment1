# exercise 1.d
from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = plt.imread('images/umbrellas.jpg') # 0-255
img = img.astype(np.float64) / 255 # conversion from uint8 to float64

cutoutrInverted, cutoutgInverted, cutoutbInverted = 1 - img[130:260, 240:450, 0], 1 - img[130:260, 240:450, 1], 1 - img[130:260, 240:450, 2]
img[130:260, 240:450, 0], img[130:260, 240:450, 1], img[130:260, 240:450, 2] = cutoutrInverted, cutoutgInverted, cutoutbInverted

f = plt.figure()
plt.imshow(img)
plt.title("invertedCutout")
plt.show()

# How is inverting a grayscale value defined for uint8?
# 255 - originalValue
