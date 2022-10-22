# exercise 2.d
from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt

def rgb2gray(rgb):
    rgb = rgb.astype(np.float64)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    v = (r + g + b) / 3
    return v.astype(np.uint8)
    
def myhist(arr, n):
    min = np.min(arr)
    max = np.max(arr)
    diff = max - min
    arr = arr.reshape(-1)
    arr = ((arr.astype(np.float64) - min) / diff) * (n-1)
    arr = arr.astype(np.uint8)
    counts = np.zeros(n, np.uint32)
    for x in arr:
        counts[x] += 1
    return counts / len(arr)


img = plt.imread('images/img0.jpg') # 0-255
img1 = plt.imread('images/img1.jpg') # 0-255
img2 = plt.imread('images/img2.jpg') # 0-255
grey = rgb2gray(np.copy(img))
grey1 = rgb2gray(np.copy(img1))
grey2 = rgb2gray(np.copy(img2))

numOfBins = 50
histogram = myhist(np.copy(grey), numOfBins)
histogram1 = myhist(np.copy(grey1), numOfBins)
histogram2 = myhist(np.copy(grey2), numOfBins)

f = plt.figure()
f.add_subplot(2, 3, 1)
plt.imshow(grey, cmap="gray")
f.add_subplot(2, 3, 2)
plt.imshow(grey1, cmap="gray")
f.add_subplot(2, 3, 3)
plt.imshow(grey2, cmap="gray")

f.add_subplot(2, 3, 4)
plt.bar(np.arange(numOfBins), histogram)
f.add_subplot(2, 3, 5)
plt.bar(np.arange(numOfBins), histogram1)
f.add_subplot(2, 3, 6)
plt.bar(np.arange(numOfBins), histogram2)

plt.show()

