# exercise 2.c
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
    
def myhist(arr, n):
    arr = arr.reshape(-1)
    arr = (arr.astype(np.float64) / 255) * (n-1)
    arr = arr.astype(np.uint8)
    counts = np.zeros(n, np.uint32)
    for x in arr:
        counts[x] += 1
    return counts / len(arr)

def myhist2(arr, n):
    min = np.min(arr)
    max = np.max(arr)
    diff = max - min
    print(min, max)    
    arr = arr.reshape(-1)
    arr = ((arr.astype(np.float64) - min) / diff) * (n-1)
    arr = arr.astype(np.uint8)
    counts = np.zeros(n, np.uint32)
    for x in arr:
        counts[x] += 1
    return counts / len(arr)


img = plt.imread('images/bird.jpg') # 0-255
grey = rgb2gray(np.copy(img))
grey = grey / 2 + 125

f = plt.figure()
f.add_subplot(2, 2, 1)
plt.imshow(grey, cmap="gray")

numOfBins = 100
histogram = myhist(np.copy(grey), numOfBins)
# print(histogram, len(histogram))

f.add_subplot(2, 2, 2)
plt.bar(np.arange(numOfBins), histogram)


numOfBins = 100
histogram = myhist2(np.copy(grey), numOfBins)
# print(histogram, len(histogram))

f.add_subplot(2, 2, 3)
plt.bar(np.arange(numOfBins), histogram)

plt.show()

