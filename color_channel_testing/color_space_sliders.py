import numpy as np
import cv2 as cv
import copy
import json
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


# def red_slider(intensity):
#     modified_image[:, :, 2] = image[:, :, 2] * (intensity / 255.0)
#     cv.imshow(window_name, modified_image)
#
#
# def green_slider(intensity):
#     modified_image[:, :, 1] = image[:, :, 1] * (intensity / 255.0)
#     cv.imshow(window_name, modified_image)
#
#
# def blue_slider(intensity):
#     modified_image[:, :, 0] = image[:, :, 0] * (intensity / 255.0)
#     cv.imshow(window_name, modified_image)

def separate_hls(rgb_img):
    hls = cv.cvtColor(rgb_img, cv.COLOR_RGB2HLS)
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]
    return h, l, s


def separate_lab(rgb_img):
    lab = cv.cvtColor(rgb_img, cv.COLOR_RGB2Lab)
    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    return l, a, b


def separate_luv(rgb_img):
    luv = cv.cvtColor(rgb_img, cv.COLOR_BGR2Luv)
    l = luv[:, :, 0]
    u = luv[:, :, 1]
    v = luv[:, :, 2]
    return l, u, v


def binary_threshold_lab_luv(rgb_img, bthresh, lthresh):
    l, a, b = separate_lab(rgb_img)
    l2, u, v = separate_luv(rgb_img)
    binary = np.zeros_like(l)
    binary[
        ((b > bthresh[0]) & (b <= bthresh[1])) |
        ((l2 > lthresh[0]) & (l2 <= lthresh[1]))
        ] = 1
    return binary


def binary_threshold_hls(rgb_img, sthresh, lthresh):
    h, l, s = separate_hls(rgb_img)
    binary = np.zeros_like(h)
    binary[
        ((s > sthresh[0]) & (s <= sthresh[1])) &
        ((l > lthresh[0]) & (l <= lthresh[1]))
        ] = 1
    return binary


def gradient_threshold(channel, thresh):
    # Take the derivative in x
    sobelx = cv.Sobel(channel, cv.CV_64F, 1, 0)
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    # Threshold gradient channel
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary

try:
    with open('thresh.json') as f:
        thresh_dict = json.loads(f.read())
        gradient_thresh = (thresh_dict['gradient_min'], thresh_dict['gradient_max'])
        s_channel_thresh = (thresh_dict['s_channel_min'], thresh_dict['s_channel_max'])
        l_channel_thresh = (thresh_dict['l_channel_min'], thresh_dict['l_channel_max'])
        b_channel_thresh = (thresh_dict['b_channel_min'], thresh_dict['b_channel_max'])
        l2_channel_thresh = (thresh_dict['l2_channel_max'], thresh_dict['l2_channel_min'])
except:
    gradient_thresh = (20, 100)
    s_channel_thresh = (80, 255)
    l_channel_thresh = (80, 255)
    b_channel_thresh = (150, 200)
    l2_channel_thresh = (225, 255)

window_name = "Test image"


def update_image():
    global image
    global gradient_thresh
    global s_channel_thresh
    global l_channel_thresh
    global b_channel_thresh
    global l2_channel_thresh
    s_binary = binary_threshold_lab_luv(image, b_channel_thresh, l2_channel_thresh)

    h, l, s = separate_hls(image)
    sxbinary = gradient_threshold(s, gradient_thresh)

    color_binary = np.dstack((sxbinary, s_binary, np.zeros_like(sxbinary))) * 255
    cv.imshow(window_name, color_binary)


def gradient_min_func(intensity):
    global gradient_thresh
    gradient_thresh = (intensity, gradient_thresh[1])
    if intensity > gradient_thresh[1]:
        cv.setTrackbarPos("Maximum Gradient", window_name, intensity)
    update_image()


def gradient_max_func(intensity):
    global gradient_thresh
    gradient_thresh = (gradient_thresh[0], intensity)
    if intensity < gradient_thresh[0]:
        cv.setTrackbarPos("Minimum Gradient", window_name, intensity)
    update_image()


def s_min_func(intensity):
    global s_channel_thresh
    s_channel_thresh = (intensity, s_channel_thresh[1])
    if intensity > s_channel_thresh[1]:
        cv.setTrackbarPos("Maximum S Channel", window_name, intensity)
    update_image()


def s_max_func(intensity):
    global s_channel_thresh
    s_channel_thresh = (s_channel_thresh[0], intensity)
    if intensity < s_channel_thresh[0]:
        cv.setTrackbarPos("Minimum S Channel", window_name, intensity)
    update_image()


def l_min_func(intensity):
    global l_channel_thresh
    l_channel_thresh = (intensity, l_channel_thresh[1])
    if intensity > l_channel_thresh[1]:
        cv.setTrackbarPos("Maximum L Channel", window_name, intensity)
    update_image()


def l_max_func(intensity):
    global l_channel_thresh
    l_channel_thresh = (l_channel_thresh[0], intensity)
    if intensity < l_channel_thresh[0]:
        cv.setTrackbarPos("Minimum L Channel", window_name, intensity)
    update_image()


def b_min_func(intensity):
    global b_channel_thresh
    b_channel_thresh = (intensity, b_channel_thresh[1])
    if intensity > b_channel_thresh[1]:
        cv.setTrackbarPos("Maximum B Channel", window_name, intensity)
    update_image()


def b_max_func(intensity):
    global b_channel_thresh
    b_channel_thresh = (b_channel_thresh[0], intensity)
    if intensity < b_channel_thresh[0]:
        cv.setTrackbarPos("Minimum B Channel", window_name, intensity)
    update_image()


def l2_min_func(intensity):
    global l2_channel_thresh
    l2_channel_thresh = (intensity, l2_channel_thresh[1])
    if intensity > l2_channel_thresh[1]:
        cv.setTrackbarPos("Maximum L2 Channel", window_name, intensity)
    update_image()


def l2_max_func(intensity):
    global l2_channel_thresh
    l2_channel_thresh = (l2_channel_thresh[0], intensity)
    if intensity < l2_channel_thresh[0]:
        cv.setTrackbarPos("Minimum L2 Channel", window_name, intensity)
    update_image()


cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.createTrackbar("Minimum Gradient", window_name, gradient_thresh[0], 255, gradient_min_func)
cv.createTrackbar("Maximum Gradient", window_name, gradient_thresh[1], 255, gradient_max_func)
cv.createTrackbar("Minimum S Channel", window_name, s_channel_thresh[0], 255, s_min_func)
cv.createTrackbar("Maximum S Channel", window_name, s_channel_thresh[1], 255, s_max_func)
cv.createTrackbar("Minimum L Channel", window_name, l_channel_thresh[0], 255, l_min_func)
cv.createTrackbar("Maximum L Channel", window_name, l_channel_thresh[1], 255, l_max_func)
cv.createTrackbar("Minimum B Channel", window_name, b_channel_thresh[0], 255, b_min_func)
cv.createTrackbar("Maximum B Channel", window_name, b_channel_thresh[1], 255, b_max_func)
cv.createTrackbar("Minimum L2 Channel", window_name, l2_channel_thresh[0], 255, l2_min_func)
cv.createTrackbar("Maximum L2 Channel", window_name, l2_channel_thresh[1], 255, l2_max_func)

update_image()
cv.waitKey(0)

with open("thresh.json", 'w') as f:
    thresh_dict = {"gradient_min": gradient_thresh[0], "gradient_max": gradient_thresh[1],
        "s_channel_min": s_channel_thresh[0], "s_channel_max": s_channel_thresh[1],
        "l_channel_min": l_channel_thresh[0], "l_channel_max": l_channel_thresh[1],
        "b_channel_min": b_channel_thresh[0], "b_channel_max": b_channel_thresh[1],
        "l2_channel_min": l2_channel_thresh[0], "l2_channel_max": l2_channel_thresh[1]}
    f.write(json.dumps(thresh_dict, indent=4, separators=(',', ': '), sort_keys=True))
