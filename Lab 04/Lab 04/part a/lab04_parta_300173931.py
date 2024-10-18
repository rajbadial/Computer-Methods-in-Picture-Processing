import numpy as np
import cv2

img = cv2.imread('Picture2.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def change(value):
    threshold = cv2.inRange(hsv, np.array([value, 180, 0]), np.array([value + 10, 255, 255]))
    res = cv2.bitwise_and(img,img, mask=threshold)

    # Show the image
    cv2.imshow('Part A', res)


cv2.namedWindow('Part A - HSV')
cv2.imshow('Part A - HSV', hsv)
cv2.namedWindow('Part A')
cv2.imshow('Part A', img)
# Trackbar to change the hue
cv2.createTrackbar('Threshold', 'Part A', 0, 180, change)
cv2.waitKey(0)