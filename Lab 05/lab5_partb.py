# Name: Raj Badial
# Student #: 300173931
import cv2
import numpy as np
# Resources used: https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
# and: https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv

# Load and preprocess the image
image = cv2.imread('images/lines_target.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Detect lines with cv2.HoughLines
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

# Function to calculate intersection point of two lines given in (rho, theta) format
def get_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2

    # Convert polar coordinates to standard line form (ax + by = c)
    a1, b1 = np.cos(theta1), np.sin(theta1)
    a2, b2 = np.cos(theta2), np.sin(theta2)
    c1, c2 = rho1, rho2

    # Calculate determinant
    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        return None  # Lines are parallel

    # Calculate intersection point (x, y)
    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant
    return int(x), int(y)

# Draw each line with maximum extension and find intersections
if lines is not None:
    # Draw the lines
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

    # Find and draw intersections
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            intersection = get_intersection(lines[i][0], lines[j][0])
            if intersection:
                # Draw intersection point as a small red circle
                cv2.circle(image, intersection, 3, (0, 0, 255), -1)

# Display the result
cv2.imshow("Extended Lines with Intersections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
