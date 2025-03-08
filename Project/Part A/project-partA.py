# Student Name: Raj Badial
# Student ID: 300173931

# Resources: 
# https://lvimuth.medium.com/hand-detection-in-python-using-opencv-and-mediapipe-30c7b54f5ff4
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
# https://mediapipe.readthedocs.io/en/latest/solutions/hands.html

import cv2
import mediapipe as mp
import os

# Suppress TensorFlow logs
# Not sure why I'm getting warnings, should be fine to delete if unwanted
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Input and Output Video Paths
video_path = r"videos\videos_partA.mp4"
output_video_path = r"videos\output_video_partA.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Dictionary to store movement history
movement_history = {}

while True:
    playing, frame = cap.read()

    # End of video
    if not playing:
        break
    
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get frame dimensions
            h, w, _ = frame.shape

            # Select landmark (wrist is 0, middle of hand is 5) - middle of hand seemed cleaner
            wrist_x = int(hand_landmarks.landmark[5].x * w)
            wrist_y = int(hand_landmarks.landmark[5].y * h)

            # Update movement history
            if hand_index not in movement_history:
                movement_history[hand_index] = []  # Initializiing the history for this hand

            # Add current position to the history
            movement_history[hand_index].append((wrist_x, wrist_y))

            # Draw the trajectory
            for i in range(1, len(movement_history[hand_index])):
                # Drawing a purple line from the previous point to the current point
                cv2.line(frame,
                         movement_history[hand_index][i - 1],
                         movement_history[hand_index][i],
                         (255, 0, 255), 2)

            # Keeping the history short to avoid excessive trails
            if len(movement_history[hand_index]) > 50:  # Keeping the last 50 points
                movement_history[hand_index] = movement_history[hand_index][-50:]

            # Calculate bounding box coordinates
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box

    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow("Hand Tracking with Bounding Boxes and Movement", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()

