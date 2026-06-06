import math

# LEFT EYE LANDMARKS
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# RIGHT EYE LANDMARKS
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# BLINK VARIABLES
blink_count = 0
blink_frames = 0
blink_detected = False

# THRESHOLD
EAR_THRESHOLD = 0.23

# REQUIRED CLOSED FRAMES
CLOSED_EYES_FRAMES = 2


# DISTANCE FUNCTION
def euclidean_distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# EYE ASPECT RATIO
def eye_aspect_ratio(eye_points):

    vertical1 = euclidean_distance(
        eye_points[1],
        eye_points[5]
    )

    vertical2 = euclidean_distance(
        eye_points[2],
        eye_points[4]
    )

    horizontal = euclidean_distance(
        eye_points[0],
        eye_points[3]
    )

    ear = (
        vertical1 + vertical2
    ) / (2.0 * horizontal)

    return ear


# BLINK DETECTION
def detect_blink(face_landmarks, w, h):

    global blink_count
    global blink_frames
    global blink_detected

    # LEFT EYE POINTS
    left_eye = []

    for idx in LEFT_EYE:

        x = int(
            face_landmarks.landmark[idx].x * w
        )

        y = int(
            face_landmarks.landmark[idx].y * h
        )

        left_eye.append((x, y))

    # RIGHT EYE POINTS
    right_eye = []

    for idx in RIGHT_EYE:

        x = int(
            face_landmarks.landmark[idx].x * w
        )

        y = int(
            face_landmarks.landmark[idx].y * h
        )

        right_eye.append((x, y))

    # EAR
    left_ear = eye_aspect_ratio(left_eye)

    right_ear = eye_aspect_ratio(right_eye)

    avg_ear = (
        left_ear + right_ear
    ) / 2.0

    # =========================
    # BLINK LOGIC
    # =========================

    if avg_ear < EAR_THRESHOLD:

        blink_frames += 1

    else:

        # COUNT ONLY ONCE
        if blink_frames >= CLOSED_EYES_FRAMES:

            if not blink_detected:

                blink_count += 1

                blink_detected = True

        else:

            blink_detected = False

        blink_frames = 0

    return blink_count