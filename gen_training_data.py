#!/usr/bin/env python
import cv2
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def get_center(cnt, img):
    h,w,d = img.shape
    mask = np.zeros((h,w), np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)
    m = cv2.moments(mask, True)
    return (m['m10']/m['m00'], m['m01']/m['m00'])

def decode_tiles(img):
    # Greyscale and edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bw = cv2.Canny(gray, 0, 50, 5);

    # HACK: some thresholds for hexagon recognition
    h,w,d = img.shape
    img_area = h*w
    hex_area = img_area / 200
    cap_area = img_area / 2000

    hex = []

    contours, hierarchy = cv2.findContours(
        bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Discard contours which don't seem like large-ish hexagons
    for cnt in contours:
        # Ramer-Douglas-Peucker algorithm
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)

        if (len(cnt) == 6 and \
            cv2.contourArea(cnt) > hex_area and \
            cv2.isContourConvex(cnt)):
            # more robust: check angles for regularity ~ 120 deg.
            # does not seem to be necessary
            hex.append(cnt)

    # Crop and save
    for i,cnt in enumerate(hex):
        center = get_center(cnt, img)
        h,w,d = img.shape
        crop = np.zeros((h,w), np.uint8)

        # Find interior bounding rect (hack)
        # bb = cv2.boundingRect(cnt)
        # x,y,xlen,ylen
        # bb = (center[0] - 40, center[1] - 40, 60, 60)

        # In python just use slicing
        LETTERBOX_RAD = 30;
        y1 = center[1]-LETTERBOX_RAD; y2 = center[1]+LETTERBOX_RAD;
        x1 = center[0]-LETTERBOX_RAD; x2 = center[0]+LETTERBOX_RAD;
        crop = bw[y1:y2, x1:x2]
        
        plt.imshow(crop, interpolation='bicubic')
        plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
        plt.savefig('contour_%d.png' % i, bbox_inches='tight')


screenshot = cv2.imread(sys.argv[1], 1)
decode_tiles(screenshot)
