"""Live webcam viewer for MediaPipe hand + pose landmarks (Tasks API).

Run from the project root:
    python src/live/landmark_viewer.py

Press 'q' in the video window to quit.
The first run downloads two small model files into data/models/.
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_DIR = os.path.join("data", "models")
MODELS = {
    "hand": (
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
    ),
    "pose": (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    ),
}

# 21 hand landmarks: pairs of points to join with lines
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Upper body only: nose, shoulders, elbows, wrists, hips
POSE_POINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24]
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]


def ensure_model(key):
    """Download the model file if we don't have it yet; return its path."""
    filename, url = MODELS[key]
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, path)
    return path


def draw_landmarks(frame, landmarks, connections, color, only=None):
    """Draw points and lines. Coordinates are normalized (0-1), so scale them."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], color, 2)
    indices = only if only is not None else range(len(pts))
    for i in indices:
        cv2.circle(frame, pts[i], 4, color, -1)


def main():
    hand_model = ensure_model("hand")
    pose_model = ensure_model("pose")

    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=hand_model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )
    pose_detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=pose_model),
            running_mode=vision.RunningMode.VIDEO,
        )
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Is another app using it?")

    start = time.time()
    prev = start
    print("Running. Press 'q' in the video window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start) * 1000)  # must keep increasing

        hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)

        # Draw on the raw frame so coordinates stay correct
        for hand in hand_result.hand_landmarks:
            draw_landmarks(frame, hand, HAND_CONNECTIONS, (0, 255, 0))
        for pose in pose_result.pose_landmarks:
            draw_landmarks(frame, pose, POSE_CONNECTIONS, (255, 128, 0), only=POSE_POINTS)

        # Mirror for display only, so it feels like a mirror to the signer
        display = cv2.flip(frame, 1)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        info = f"Hands: {len(hand_result.hand_landmarks)}  Pose: {len(pose_result.pose_landmarks)}  FPS: {fps:.0f}"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("ISL landmark viewer (q to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_detector.close()
    pose_detector.close()


if __name__ == "__main__":
    main()
