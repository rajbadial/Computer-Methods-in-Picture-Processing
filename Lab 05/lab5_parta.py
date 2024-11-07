# Name: Raj Badial
# Student #: 300173931
# resource to understand HoughCircles parameters: https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
import cv2
import numpy as np

# Load the image
image = cv2.imread('images/circles_target.jpg')
grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.GaussianBlur(grayscale_img, (9, 9), 2)
circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp = 1, minRadius = 30, maxRadius = 60, minDist = 30, param1 = 50, param2 = 15)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # Drawing center of circle(s)
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        # Drawing circumference of circle(s)
        cv2.circle(image, (x, y), r, (0, 255, 0), 3)

cv2.imshow("Detected Circles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()