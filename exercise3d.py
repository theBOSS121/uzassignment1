# exercise 3.d
from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt

def rgb2gray(rgb):
    rgb = rgb.astype(np.float64)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    v = (r + g + b) / 3
    return v.astype(np.uint8)

def immask(img, mask):
    m = np.expand_dims(mask, axis=2)
    return np.where(m == 1, img, [0, 0, 0])

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

img = plt.imread('images/eagle.jpg') # 0-255
grey = rgb2gray(np.copy(img))

# threshold = 185
# mask = np.where(grey < threshold, 0, 1)
numOfBins = 256
histogram = myhist(np.copy(grey), numOfBins) # returns normalized probabilities for bins
threshold = otsu(histogram, numOfBins)
mask = np.where(grey < threshold, 0, 1) 
mask = mask.astype(np.uint8)

n = 10
# se = np.ones((n,n), np.uint8) # create a square structuring element
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))

img_eroded = cv2.erode(mask, se)
img_dilated = cv2.dilate(mask, se)

img_eroded_dilated = cv2.dilate(img_eroded, se)
img_dilated_eroded = cv2.erode(img_dilated, se)

imgMasked = immask(img, img_eroded_dilated)

f = plt.figure()
f.add_subplot(2, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Grey")

f.add_subplot(2, 3, 2)
plt.imshow(imgMasked, cmap="gray")
plt.title("Mask")

f.add_subplot(2, 3, 3)
plt.imshow(img_eroded, cmap="gray")
plt.title("Eroded")

f.add_subplot(2, 3, 4)
plt.imshow(img_dilated, cmap="gray")
plt.title("Dilated")

f.add_subplot(2, 3, 5)
plt.imshow(img_eroded_dilated, cmap="gray")
plt.title("Eroded-dilated")

f.add_subplot(2, 3, 6)
plt.imshow(img_dilated_eroded, cmap="gray")
plt.title("Dilated-eroded")

plt.show()

# Question: Why is the background included in the mask and not the object? How would you fix that in general? (just inverting the mask if necessary doesn’t count)
# Background is lighter than the object. Use a mehod to substact a background (detect what is foreground/background) => adjust algoithm
