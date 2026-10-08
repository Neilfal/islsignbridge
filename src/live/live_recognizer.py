"""Live ISL word recognizer (key-press mode).

Run from anywhere inside the project:
    python src/live/live_recognizer.py
    python src/live/live_recognizer.py --speak        # also speak the word (needs pyttsx3)
    python src/live/live_recognizer.py --threshold 0.5

How to use:
    SPACE  start: 3-second countdown, then it records your sign
    SPACE  again to stop recording (or it stops by itself after 5 seconds)
    q      quit

Tips: sit or stand far enough back that both shoulders AND both hands are in
view (the training videos show people standing back from the camera). Good
lighting and a plain background help.

Needs: data/models/isl_rf.joblib (python src/models/train_final.py) and the
MediaPipe model files in data/models/ (downloaded by landmark_viewer.py).
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.build_features import clip_features, resample  # noqa: E402

MODEL_DIR = ROOT / "data" / "models"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"
CLASSIFIER = MODEL_DIR / "isl_rf.joblib"

COUNTDOWN_S = 3.0
MAX_RECORD_S = 5.0
MIN_FRAMES = 10
MIN_HAND_FRACTION = 0.3
PROCESS_WIDTH = 640  # same width the training videos were processed at

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
POSE_POINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24]
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]


def create_detectors():
    import mediapipe as mp  # imported here so the rest of the file can be tested without it
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    for m in (HAND_MODEL, POSE_MODEL):
        if not m.exists():
            raise FileNotFoundError(f"Missing {m}. Run src/live/landmark_viewer.py once to download it.")
    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
        )
    )
    return mp, hand, pose


def to_arrays(hand_res, pose_res):
    """Same slot logic as extract_landmarks.py, so live input matches the training data."""
    hands = np.full((2, 21, 3), np.nan, dtype=np.float32)
    for lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
        slot = 0 if handed[0].category_name == "Left" else 1
        if not np.isnan(hands[slot, 0, 0]):
            slot = 1 - slot
        hands[slot] = [[lm.x, lm.y, lm.z] for lm in lms]
    pose = np.full((33, 4), np.nan, dtype=np.float32)
    if pose_res.pose_landmarks:
        pose[:] = [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_res.pose_landmarks[0]]
    return hands, pose


def recognize(bundle, hands_list, pose_list, aspect, threshold):
    """Turn a recorded clip into a prediction. Returns a dict for the screen."""
    if len(hands_list) < MIN_FRAMES:
        return {"error": "Recording too short. Try again."}
    H, P = np.stack(hands_list), np.stack(pose_list)
    seen = (~np.isnan(H[:, :, 0, 0])).any(axis=1).mean()
    if seen < MIN_HAND_FRACTION:
        return {"error": "Hands were not visible. Step back and try again."}

    feats = resample(clip_features(H, P, aspect=aspect), bundle["n_frames"])
    proba = bundle["model"].predict_proba(feats.reshape(1, -1))[0]
    classes = bundle["model"].classes_
    order = np.argsort(proba)[::-1]
    top = [(bundle["names"][int(classes[i])], float(proba[i])) for i in order[:3]]
    return {
        "word": top[0][0],
        "confidence": top[0][1],
        "confident": top[0][1] >= threshold,
        "top3": top,
        "hand_fraction": float(seen),
    }


def draw_landmarks(frame, lms, connections, color, only=None):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for i in (only if only is not None else range(len(pts))):
        cv2.circle(frame, pts[i], 4, color, -1)


def put_text(img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.40,
                        help="below this confidence the app says 'not sure' (not calibrated, tune on your own data)")
    parser.add_argument("--speak", action="store_true", help="speak the recognized word")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if not CLASSIFIER.exists():
        raise FileNotFoundError(f"Missing {CLASSIFIER}. Run: python src/models/train_final.py")
    bundle = joblib.load(CLASSIFIER)

    engine = None
    if args.speak:
        try:
            import pyttsx3
            engine = pyttsx3.init()
        except Exception as e:
            print(f"Speech is not available ({e}). Install it with: pip install pyttsx3")

    mp, hand_det, pose_det = create_detectors()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Is another app using it?")

    state, t_start, rec_t0 = "idle", 0.0, 0.0
    rec_hands, rec_pose = [], []
    result = None
    start, last_ts = time.time(), -1
    print("Ready. SPACE = start/stop, q = quit.")

    def finish(aspect):
        nonlocal state, result
        result = recognize(bundle, rec_hands, rec_pose, aspect, args.threshold)
        state = "idle"
        if "word" in result:
            print(f"{result['word']} ({result['confidence']:.0%})  top3={result['top3']}")
            if engine is not None and result["confident"]:
                engine.say(result["word"])
                engine.runAndWait()
        else:
            print(result["error"])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h0, w0 = frame.shape[:2]
        aspect = w0 / h0
        if w0 > PROCESS_WIDTH:
            frame = cv2.resize(frame, (PROCESS_WIDTH, int(h0 * PROCESS_WIDTH / w0)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int((time.time() - start) * 1000), last_ts + 1)
        last_ts = ts
        hand_res = hand_det.detect_for_video(mp_image, ts)
        pose_res = pose_det.detect_for_video(mp_image, ts)

        now = time.time()
        if state == "countdown" and now >= t_start:
            state, rec_t0 = "recording", now
            rec_hands, rec_pose = [], []
        if state == "recording":
            hands, pose = to_arrays(hand_res, pose_res)
            rec_hands.append(hands)
            rec_pose.append(pose)
            if now - rec_t0 >= MAX_RECORD_S:
                finish(aspect)

        # Draw on the raw frame, then mirror for display so it feels like a mirror
        for hand in hand_res.hand_landmarks:
            draw_landmarks(frame, hand, HAND_CONNECTIONS, (0, 255, 0))
        for pose in pose_res.pose_landmarks:
            draw_landmarks(frame, pose, POSE_CONNECTIONS, (255, 128, 0), only=POSE_POINTS)
        display = cv2.flip(frame, 1)
        h, w = display.shape[:2]

        if state == "idle":
            put_text(display, "SPACE: start (3 s countdown)   q: quit", (10, 28), 0.6)
        elif state == "countdown":
            put_text(display, "Get ready... both shoulders and hands in view", (10, 28), 0.6)
            put_text(display, str(int(np.ceil(t_start - now))), (w // 2 - 20, h // 2 + 30), 3.0, (0, 255, 255), 6)
        else:
            cv2.circle(display, (20, 22), 9, (0, 0, 255), -1)
            put_text(display, f"RECORDING {now - rec_t0:.1f}s  (SPACE to stop)", (38, 28), 0.6, (0, 0, 255))

        if result is not None and state == "idle":
            bar = display.copy()
            cv2.rectangle(bar, (0, h - 110), (w, h), (0, 0, 0), -1)
            display = cv2.addWeighted(bar, 0.6, display, 0.4, 0)
            if "error" in result:
                put_text(display, result["error"], (10, h - 60), 0.7, (0, 165, 255))
            elif result["confident"]:
                put_text(display, f"{result['word'].upper()}  {result['confidence']:.0%}", (10, h - 62), 1.4, (0, 255, 0), 3)
            else:
                put_text(display, f"Not sure (best guess {result['word']} {result['confidence']:.0%})",
                         (10, h - 62), 0.8, (0, 165, 255))
            if "top3" in result:
                alt = "   ".join(f"{n} {p:.0%}" for n, p in result["top3"])
                put_text(display, alt, (10, h - 22), 0.55)

        cv2.imshow("ISL recognizer", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32:  # SPACE
            if state == "idle":
                state, t_start, result = "countdown", time.time() + COUNTDOWN_S, None
            elif state == "recording":
                finish(aspect)
            elif state == "countdown":
                state = "idle"

    cap.release()
    cv2.destroyAllWindows()
    hand_det.close()
    pose_det.close()


if __name__ == "__main__":
    main()
