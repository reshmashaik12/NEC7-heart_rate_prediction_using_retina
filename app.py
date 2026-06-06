from flask import Flask, render_template, Response, jsonify
import cv2

from utils.eye_detection import detect_eyes
from utils.eye_roi import draw_eye_box
from utils.blink_detection import detect_blink
from utils.heart_rate import calculate_heart_rate

app = Flask(__name__)

camera = cv2.VideoCapture(0)

# LIVE VALUES
live_blink = 0
live_bpm = 72
live_accuracy = 95

def generate_frames():

    global live_blink
    global live_bpm
    global live_accuracy

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        results = detect_eyes(frame)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                # Eye Box
                frame = draw_eye_box(
                    frame,
                    face_landmarks
                )

                # Blink Detection
                blink_count = detect_blink(
                    face_landmarks,
                    w,
                    h
                )

                # Heart Rate
                bpm = calculate_heart_rate(
                    blink_count
                )

                # Update Live Values
                live_blink = blink_count
                live_bpm = bpm
                live_accuracy = 95

        # Encode Frame
        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

@app.route('/')

def index():

    return render_template('index.html')

@app.route('/video')

def video():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# SEND LIVE DATA
@app.route('/data')

def data():

    return jsonify({
        "blink": live_blink,
        "bpm": live_bpm,
        "accuracy": live_accuracy
    })

if __name__ == "__main__":

    app.run(debug=True)