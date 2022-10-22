# exercise 3.a and 3.b
from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = plt.imread('images/mask.png') # 0-255

n = 4
se = np.ones((n,n), np.uint8) # create a square structuring element
# se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))

img_eroded = cv2.erode(img, se)
img_dilated = cv2.dilate(img, se)

img_eroded_dilated = cv2.dilate(img_eroded, se)
img_dilated_eroded = cv2.erode(img_dilated, se)

f = plt.figure()
f.add_subplot(2, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Grey")

f.add_subplot(2, 3, 2)
plt.imshow(img_eroded, cmap="gray")
plt.title("Eroded")

f.add_subplot(2, 3, 3)
plt.imshow(img_dilated, cmap="gray")
plt.title("Dilated")

f.add_subplot(2, 3, 4)
plt.imshow(img_eroded_dilated, cmap="gray")
plt.title("Eroded-dilated")

f.add_subplot(2, 3, 5)
plt.imshow(img_dilated_eroded, cmap="gray")
plt.title("Dilated-eroded")

plt.show()

# Question: Based on the results, which order of erosion and dilation operations produces opening and which closing?
# erosion => dilation = opening
# dilation => erosion = closing