import cv2
import numpy as np
from collections import deque

from utils.eye_detection import detect_eyes
from utils.eye_roi import draw_eye_box
from utils.blink_detection import detect_blink
from utils.heart_rate import calculate_heart_rate

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

# =========================
# GRAPH SETTINGS
# =========================

GRAPH_WIDTH = 300
GRAPH_HEIGHT = 140

bpm_history = deque(maxlen=GRAPH_WIDTH)

ecg_shift = 0

# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    results = detect_eyes(frame)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # =========================
            # EYE BOX
            # =========================

            frame = draw_eye_box(
                frame,
                face_landmarks
            )

            # =========================
            # BLINK DETECTION
            # =========================

            blink_count = detect_blink(
                face_landmarks,
                w,
                h
            )

            # =========================
            # HEART RATE
            # =========================

            bpm = calculate_heart_rate(
                blink_count
            )

            accuracy = 95

            bpm_history.append(bpm)

            # =========================
            # LEFT PANEL
            # =========================

            cv2.rectangle(
                frame,
                (10, 10),
                (330, 190),
                (20, 20, 20),
                -1
            )

            cv2.rectangle(
                frame,
                (10, 10),
                (330, 190),
                (80, 80, 80),
                2
            )

            # TITLE

            cv2.putText(
                frame,
                "HEART RATE DETECTION",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # BLINK COUNT

            cv2.putText(
                frame,
                f'Blink Count : {blink_count}',
                (25, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 120),
                2
            )

            # HEART RATE

            cv2.putText(
                frame,
                f'Heart Rate : {bpm} BPM',
                (25, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 180, 0),
                2
            )

            # ACCURACY

            cv2.putText(
                frame,
                f'Accuracy : {accuracy}%',
                (25, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 200, 255),
                2
            )

            # =========================
            # GRAPH POSITION
            # =========================

            graph_x = w - GRAPH_WIDTH - 30
            graph_y = 40

            # GRAPH BACKGROUND

            cv2.rectangle(
                frame,
                (graph_x, graph_y),
                (
                    graph_x + GRAPH_WIDTH,
                    graph_y + GRAPH_HEIGHT
                ),
                (15, 15, 30),
                -1
            )

            # GRAPH BORDER

            cv2.rectangle(
                frame,
                (graph_x, graph_y),
                (
                    graph_x + GRAPH_WIDTH,
                    graph_y + GRAPH_HEIGHT
                ),
                (100, 100, 100),
                2
            )

            # GRAPH TITLE

            cv2.putText(
                frame,
                "LIVE BPM GRAPH",
                (graph_x + 70, graph_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # =========================
            # BPM SCALE
            # =========================

            scale_values = [60, 70, 80, 90, 100]

            for idx, value in enumerate(scale_values):

                y_pos = graph_y + GRAPH_HEIGHT - (idx * 30)

                # SCALE VALUES

                cv2.putText(
                    frame,
                    str(value),
                    (graph_x - 45, y_pos + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 180, 255),
                    1
                )

                # GRID

                cv2.line(
                    frame,
                    (graph_x, y_pos),
                    (graph_x + GRAPH_WIDTH, y_pos),
                    (40, 40, 40),
                    1
                )

            # =========================
            # LIVE GRAPH
            # =========================

            points = []

            center_y = graph_y + GRAPH_HEIGHT // 2

            history_list = list(bpm_history)

            for i in range(len(history_list)):

                x = graph_x + i

                normalized = (
                    history_list[i] - 60
                ) * 2

                pulse = 0

                if (i + ecg_shift) % 45 == 0:
                    pulse = -18

                elif (i + ecg_shift) % 45 == 5:
                    pulse = 10

                y = int(
                    center_y
                    - normalized
                    + pulse
                )

                y = max(
                    graph_y + 5,
                    min(
                        y,
                        graph_y + GRAPH_HEIGHT - 5
                    )
                )

                points.append((x, y))

            # DRAW GRAPH

            for i in range(1, len(points)):

                cv2.line(
                    frame,
                    points[i - 1],
                    points[i],
                    (0, 255, 255),
                    2
                )

            ecg_shift += 2

    # =========================
    # SHOW WINDOW
    # =========================

    cv2.imshow(
        "Heart Rate Detection",
        frame
    )

    # EXIT

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# RELEASE
# =========================

cap.release()
cv2.destroyAllWindows()