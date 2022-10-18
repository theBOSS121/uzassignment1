# exercise 1.a
from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt

I = imread('images/umbrellas.jpg') # 0-1
# I = cv2.imread('images/umbrellas.jpg') # different colors bgr
# I = plt.imread('images/umbrellas.jpg') # 0-255
# I = I.astype(np.float64) / 255 # conversion from uint8 to float64
print(I)
print(f"ndim: {I.ndim}, size: {I.size}, shape (height, width, channels): {I.shape}, type: {I.dtype}")
# I_new=I # this is only a reference
# I_new=np.copy(I) # this copies data

imshow(I)

