import cv2
import numpy as np

LEFT_EYE = [33, 133, 160, 159, 158, 144]
RIGHT_EYE = [362, 263, 387, 386, 385, 373]

def draw_eye_box(frame, landmarks):

    h, w, _ = frame.shape

    all_points = []

    # LEFT EYE
    for idx in LEFT_EYE:

        x = int(landmarks.landmark[idx].x * w)
        y = int(landmarks.landmark[idx].y * h)

        all_points.append((x, y))

    # RIGHT EYE
    for idx in RIGHT_EYE:

        x = int(landmarks.landmark[idx].x * w)
        y = int(landmarks.landmark[idx].y * h)

        all_points.append((x, y))

    points = np.array(all_points)

    x, y, w_box, h_box = cv2.boundingRect(points)

    padding = 15

    x = x - padding
    y = y - padding

    w_box = w_box + (padding * 2)
    h_box = h_box + (padding * 2)

    cv2.rectangle(
        frame,
        (x, y),
        (x + w_box, y + h_box),
        (0, 255, 0),
        2
    )

    return frame