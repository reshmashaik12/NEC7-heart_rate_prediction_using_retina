import cv2
from mediapipe.python.solutions import face_mesh

mp_face_mesh = face_mesh

faceMesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def detect_eyes(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(rgb)

    return results