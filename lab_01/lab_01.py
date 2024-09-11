# Rajvir Badial - 300173931 - CSI 4133 Lab 01 - 2024-09-11

import cv2
import numpy as np

# Image path for the field.jpg given in the Lab 01 folder
image_path = "field.jpg"

# Read in the image into a variable to be used for other cv2 functions
read_img = cv2.imread(image_path)
write_img = cv2.imwrite("copy_of_original.png", read_img)

# Show the original image to the user for ~8 seconds
show_img = cv2.imshow("Original image", read_img)
cv2.waitKey(7000)

# Downsize twice then upsize the image twice to show the loss of quality when downsizing by a factor of 4
downsize_img = cv2.pyrDown(read_img)
downsize_img = cv2.pyrDown(downsize_img)
downsize_img = cv2.pyrUp(downsize_img)
downsize_img = cv2.pyrUp(downsize_img)
# could also do cv2.resize(read_img, (0,0), fx=0.25, fy=0.25), then cv2.resize(read_img, (0,0), fx=4, fy=4)

# Show the downsized image to the user for ~8 seconds
cv2.imshow("Downsized Image 4x", downsize_img)
cv2.waitKey(7000)


# Quantizing image, aid taken from: https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html
Z = read_img.reshape((-1,3))

# define criteria, number of clusters(K) and apply kmeans()
ret,label,center = cv2.kmeans(
	np.float32(Z), # Data to quantize
	32, # Quantize to 32 levels
	None,
	(cv2.TERM_CRITERIA_MAX_ITER, 3, 1),
	3, # 3 iterations max
	cv2.KMEANS_RANDOM_CENTERS
)

center = np.uint8(center)
res = center[label.flatten()]
quantized_img = res.reshape((read_img.shape))

cv2.imshow('Quantized Image', quantized_img)
cv2.waitKey(7000)

write_img = cv2.imwrite("downsized.jpg", downsize_img)
write_img = cv2.imwrite("quantized.jpg", quantized_img)

# Close all windows (original, downsized and quantized windows)
cv2.destroyAllWindows()