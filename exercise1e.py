# exercise 1.e
from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = imread_gray("images/umbrellas.jpg")
imgReduced = np.copy(img)
imgReduced *= 63
imgReduced = imgReduced.astype(np.uint8)

f = plt.figure()
f.add_subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("original (gray)")
f.add_subplot(2, 2, 2)
plt.imshow(imgReduced, cmap="gray")
plt.title("reduced")
f.add_subplot(2, 2, 3)
plt.imshow(imgReduced, cmap="gray", vmax=255)
plt.title("reduced, vmax=255")
plt.show()