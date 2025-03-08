'''
Counts the number of extended fingers. Thumb is only considered from the horizontal direction compared
to the other fingers

Resources used were my Part A code and # https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
for knowing the location of points in the hand_lanmarks variable
'''
def count_extended_fingers(hand_landmarks, handedness):
    # Finger tip and base landmark indices
    # Thumb, Index, Middle, Ring, Pinky
    finger_tips = [4, 8, 12, 16, 20]
    finger_bases = [3, 5, 9, 13, 17]

    extended_count = 0

    for tip, base in zip(finger_tips, finger_bases):
        # Thumb requires special treatment
        if tip == 4:  # Thumb
            thumb_tip = hand_landmarks.landmark[tip]
            thumb_base = hand_landmarks.landmark[base]

            # Check if the thumb is horizontally extended
            # Adjusted thumb logic based on handedness
            if handedness.classification[0].label == "Left":
                #print("Left HANd!!") - was used for debugging
                if thumb_tip.x > thumb_base.x:  # Thumb extends rightwards for left hand
                    extended_count += 1
            else:  # "Right"
                #print("right HANd!!")
                if thumb_tip.x < thumb_base.x:  # Thumb extends leftwards for right hand
                    extended_count += 1
        else:
            # Other fingers are extended if tip is higher than the base
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
                extended_count += 1

    return extended_count


def detect_numbers(results):
    if not results.multi_hand_landmarks:
        return 0  # No hands detected

    # Detect the number of extended fingers for each hand
    hand_counts = []
    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        hand_counts.append(count_extended_fingers(hand_landmarks, handedness))

    # Combine counts from both hands
    return sum(hand_counts)


import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)  # Uses webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Detect numbers
    number = detect_numbers(results)

    # Draw landmarks and number
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Displays the detected number
    cv2.putText(frame, f"Number: {number}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Number Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
