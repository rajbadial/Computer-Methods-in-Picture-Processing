import numpy as np
import cv2

# load and show the image
img = cv2.imread('Picture3.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('Part B', hsv)

# assign the colour values
yellow = [np.array([30, 100, 100]), np.array([40, 255, 255])]
violet = [np.array([125, 70, 70]), np.array([130, 255, 255])]
red = [np.array([0, 150, 150]), np.array([0, 255, 255])]

def change(value):
    arr = yellow
    if value == 1:
        arr = violet
    elif value == 2:
        arr = red

    threshold = cv2.inRange(hsv, arr[0], arr[1])

    res = cv2.bitwise_and(img,img, mask=threshold)

    # Show the image
    cv2.imshow('Part B - mask', threshold)
    cv2.imshow('Part B - result', res)

cv2.namedWindow('Part B - mask')
cv2.namedWindow('Part B - result')
cv2.createTrackbar('Threshold', 'Part B - result', 0, 2, change)
cv2.waitKey(0)