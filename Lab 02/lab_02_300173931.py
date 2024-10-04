import cv2

# Load the consecutive images
image1 = cv2.imread("images01\Img02_0076.bmp")
image2 = cv2.imread("images01\Img02_0077.bmp")
image3 = cv2.imread("images01\Img02_0078.bmp")
image4 = cv2.imread("images01\park466.bmp")
image5 = cv2.imread("images01\park467.bmp")
image6 = cv2.imread("images01\park468.bmp")

# Convert the images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
gray4 = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
gray5 = cv2.cvtColor(image5, cv2.COLOR_BGR2GRAY)
gray6 = cv2.cvtColor(image6, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Motion Detection')

# Function to update threshold dynamically
def update_threshold(x):
    # Calculate the absolute difference between the grayscale images
    diff_im1_im2 = cv2.absdiff(gray1, gray2)
    diff_im2_im3 = cv2.absdiff(gray2, gray3)
    combined_diff1 = cv2.bitwise_or(diff_im1_im2, diff_im2_im3)

    # Apply thresholding
    _, new_img1 = cv2.threshold(combined_diff1, x, 255, cv2.THRESH_BINARY)

    # Display the resulting image
    cv2.imshow('Motion Detection', new_img1)

    # Save the binary result image
    cv2.imwrite("motion_result.png", new_img1)

# Create a track bar to adjust the threshold value dynamically
cv2.createTrackbar('Threshold', 'Motion Detection', 0, 255, update_threshold)
cv2.setTrackbarPos('Threshold', 'Motion Detection', 30)

cv2.waitKey(0)
cv2.destroyWindow('Motion Detection')
cv2.namedWindow('Runner')

# exact same as other function, just for the image(s) of the runner
def update_threshold_2(x):
    diff_im4_im5 = cv2.absdiff(gray4, gray5)
    diff_im5_im6 = cv2.absdiff(gray5, gray6)
    combined_diff2 = cv2.bitwise_or(diff_im4_im5, diff_im5_im6)

    _, new_img2 = cv2.threshold(combined_diff2, x, 255, cv2.THRESH_BINARY)

    cv2.imshow('Runner', new_img2)
    cv2.imwrite('runner_result.png', new_img2)

cv2.createTrackbar('Threshold', 'Runner', 0, 255, update_threshold_2)
cv2.setTrackbarPos('Threshold', 'Runner', 30)

cv2.waitKey(0)
cv2.destroyAllWindows()