# exercise 2.e
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

def otsu(hist, numOfBins):
    weight1 = np.cumsum(hist) # probabilities of class 1
    weight2 = np.cumsum(hist[::-1])[::-1] # probabilities of class 1
    binMids = np.arange(numOfBins) # makes array [0, 1, 2,..., numOfBins-1]
    mean1 = np.cumsum(hist * binMids) / weight1
    mean2 = (np.cumsum((hist * binMids)[::-1]) / weight2[::-1])[::-1]
    interClassVariance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    return np.argmax(interClassVariance) # max of interClassVariance function value


img = plt.imread('images/bird.jpg') # 0-255
grey = rgb2gray(np.copy(img))
numOfBins = 256
histogram = myhist(np.copy(grey), numOfBins) # returns normalized probabilities for bins
threshold = otsu(histogram, numOfBins)
print(threshold)
mask = np.where(grey < threshold, 0, 1) 

f = plt.figure()
f.add_subplot(2, 2, 1)
plt.imshow(img)
f.add_subplot(2, 2, 2)
plt.imshow(grey, cmap="gray")
f.add_subplot(2, 2, 3)
plt.bar(np.arange(numOfBins), histogram)
f.add_subplot(2, 2, 4)
plt.imshow(mask, cmap="gray")

plt.show()
