# exercise 2.e
from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt

def otsu(img):
    # testing all thresholds from 0 to the maximum of the image
    threshold_range = range(np.max(256))
    criterias = [compute_otsu_criteria(img, th) for th in threshold_range]
    # best threshold is the one minimizing the Otsu criteria
    return threshold_range[np.argmin(criterias)]

def compute_otsu_criteria(im, th):
    # create the thresholded image
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1
    # compute weights
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1
    # if one the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf
    # find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]
    # compute variance of these classes
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    return weight0 * var0 + weight1 * var1

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


img = plt.imread('images/bird.jpg') # 0-255
grey = rgb2gray(np.copy(img))

numOfBins = 255
histogram = myhist(np.copy(grey), numOfBins)
threshold = otsu(grey)
mask = np.where(grey < threshold, 0, 1) 
print(threshold)


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
