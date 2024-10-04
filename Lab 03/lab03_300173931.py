import cv2
import numpy as np

# Obtaining video into a variable, setting the output
capture = cv2.VideoCapture('./video/park.avi')

# Had issues with output not being saved. Was because isColor had to be set to False (output was expecting a BGR video)
output = cv2.VideoWriter('./output/park.avi', cv2.VideoWriter_fourcc(*'XVID'), 1, (320,240), isColor=False)

# Getting first frame and converting it to grayscale
_, first_frame = capture.read()
first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = capture.read()
    if not ret:
        break
    
    # convert next frame to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate difference between the frames
    diff = cv2.absdiff(first_frame_gray, frame)

    # spcify the threshold we're looking for
    _, thresh_image = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # switch to next frame (sorry naming conventions may not make much sense - first_frame_gray is 
    # the previous frame, frame is the current frame)
    first_frame_gray = frame
    output.write(thresh_image)

    # Output to the screen
    cv2.imshow('Video', thresh_image)
    if cv2.waitKey(24) & 0xFF == ord('q'):
        break

capture.release()
output.release()
cv2.destroyAllWindows()