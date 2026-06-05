import numpy as np

blink_count = 0
blink_detected = False

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calculate_distance(p1, p2):

    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye_points):

    vertical1 = calculate_distance(eye_points[1], eye_points[5])
    vertical2 = calculate_distance(eye_points[2], eye_points[4])

    horizontal = calculate_distance(eye_points[0], eye_points[3])

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear

def detect_blink(face_landmarks, w, h):

    global blink_count
    global blink_detected

    left_eye = []
    right_eye = []

    for idx in LEFT_EYE:

        x = int(face_landmarks.landmark[idx].x * w)
        y = int(face_landmarks.landmark[idx].y * h)

        left_eye.append((x, y))

    for idx in RIGHT_EYE:

        x = int(face_landmarks.landmark[idx].x * w)
        y = int(face_landmarks.landmark[idx].y * h)

        right_eye.append((x, y))

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    ear = (left_ear + right_ear) / 2

    if ear < 0.20:

        if not blink_detected:
            blink_count += 1
            blink_detected = True

    else:
        blink_detected = False

    return blink_count