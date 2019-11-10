import numpy as np
import cv2 as cv
import copy
import sys
import time

if len(sys.argv) != 2:
    print("usage: " + sys.argv[0] + " <image_name>")
    exit(-1)

image = cv.imread(sys.argv[1], flags=cv.IMREAD_COLOR)
if image is None:
    print("Could not open or find the image.")
    exit(-1)
image = image[:, 0:1279]
modified_image = copy.deepcopy(image)

def red_slider(intensity):
    modified_image[:,:,2] = image[:,:,2] * (intensity/255.0)
    cv.imshow("Test image", modified_image)

def green_slider(intensity):
    modified_image[:,:,1] = image[:,:,1] * (intensity/255.0)
    cv.imshow("Test image", modified_image)

def blue_slider(intensity):
    modified_image[:,:,0] = image[:,:,0] * (intensity/255.0)
    cv.imshow("Test image", modified_image)

cv.namedWindow("Test image")
cv.createTrackbar("Red", "Test image", 255, 255, red_slider)
cv.createTrackbar("Green", "Test image", 255, 255, green_slider)
cv.createTrackbar("Blue", "Test image", 255, 255, blue_slider)
red_slider(255)

cv.waitKey(0)
