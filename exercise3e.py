# exercise 3.e
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


img = plt.imread('images/coins.jpg') # 0-255
img_without_big_coins = np.copy(img)
grey = rgb2gray(np.copy(img))

threshold = 225
mask = np.where(grey > threshold, 0, 1)
mask = mask.astype(np.uint8)

n = 4
se = np.ones((n,n), np.uint8) # create a square structuring element
# se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))

img_eroded = cv2.erode(mask, se)
img_dilated = cv2.dilate(mask, se)

img_eroded_dilated = cv2.dilate(img_eroded, se)
img_dilated_eroded = cv2.erode(img_dilated, se)

num_of_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_dilated_eroded, connectivity=8)
# print(num_of_components)
# print(output)
# print(stats)
# print(centroids)


for i in range(1, num_of_components): # go through connected components
    x = stats[i, cv2.CC_STAT_LEFT] - 5
    y = stats[i, cv2.CC_STAT_TOP] - 5
    w = stats[i, cv2.CC_STAT_WIDTH] + 10
    h = stats[i, cv2.CC_STAT_HEIGHT] + 10
    area = stats[i, cv2.CC_STAT_AREA]
    (cx, cy) = centroids[i]
    print(x, y, w, h, area)
    print(cx, cy)
    if area > 700:
        print(i)
        img_without_big_coins[y:y+h, x:x+w, 0], img_without_big_coins[y:y+h, x:x+w, 1], img_without_big_coins[y:y+h, x:x+w, 2] = 255, 255, 255




f = plt.figure()
f.add_subplot(2, 3, 1)
plt.imshow(grey, cmap="gray")
plt.title("Grey")

f.add_subplot(2, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")

f.add_subplot(2, 3, 3)
plt.imshow(img_dilated_eroded, cmap="gray")
plt.title("Dilated-eroded")

f.add_subplot(2, 3, 4)
plt.imshow(output, cmap="gray")
plt.title("output")

f.add_subplot(2, 3, 5)
plt.imshow(img, cmap="gray")
plt.title("Original")

f.add_subplot(2, 3, 6)
plt.imshow(img_without_big_coins)
plt.title("Without big coins")

plt.show()