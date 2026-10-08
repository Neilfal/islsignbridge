"""Extract MediaPipe hand + pose landmarks from INCLUDE videos.

Run from the project root:
    python src/extract/extract_landmarks.py --limit 2     # quick test (2 videos per word)
    python src/extract/extract_landmarks.py               # full run

Needs the model files in data/models/ (they are downloaded the first time you
run src/live/landmark_viewer.py).

Output (all inside data/, which Git ignores):
    data/landmarks/<class_id>_<word>/<video_name>.npz
        hands: (frames, 2, 21, 3)  x, y, z per hand landmark, NaN if no hand
        pose:  (frames, 33, 4)     x, y, z, visibility, NaN if no person
        fps:   frames per second of the video
    data/landmarks/manifest.csv    one row per video (word, file number, stats)

Re-running skips videos that are already done, so you can stop and resume.
"""

import argparse
import csv
import re
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_DIR = Path("data") / "models"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"

VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}
CLASS_DIR_RE = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")  # e.g. "48. Hello"
TARGET_WIDTH = 640  # shrink big frames: faster, and landmarks are normalized anyway


def make_detectors():
    """New detectors for every video, so tracking state never leaks between videos."""
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
    return hand, pose


def process_video(path):
    """Return (hands, pose, fps) arrays for one video, or None if it can't be read."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0

    hand_det, pose_det = make_detectors()
    hands_all, pose_all = [], []
    last_ts = -1
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > TARGET_WIDTH:
                frame = cv2.resize(frame, (TARGET_WIDTH, int(h * TARGET_WIDTH / w)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts = max(int(i * 1000 / fps), last_ts + 1)  # must strictly increase
            last_ts = ts
            i += 1

            hand_res = hand_det.detect_for_video(mp_image, ts)
            pose_res = pose_det.detect_for_video(mp_image, ts)

            # Slot 0 / slot 1 are just MediaPipe's "Left"/"Right" labels on the
            # raw (unmirrored) frame. Live webcam input is processed the same
            # way, so the slots stay consistent between training and demo.
            hands = np.full((2, 21, 3), np.nan, dtype=np.float32)
            for lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
                slot = 0 if handed[0].category_name == "Left" else 1
                if not np.isnan(hands[slot, 0, 0]):  # slot already used
                    slot = 1 - slot
                hands[slot] = [[lm.x, lm.y, lm.z] for lm in lms]

            pose = np.full((33, 4), np.nan, dtype=np.float32)
            if pose_res.pose_landmarks:
                pose[:] = [
                    [lm.x, lm.y, lm.z, lm.visibility] for lm in pose_res.pose_landmarks[0]
                ]

            hands_all.append(hands)
            pose_all.append(pose)
    finally:
        cap.release()
        hand_det.close()
        pose_det.close()

    if not hands_all:
        return None
    return np.stack(hands_all), np.stack(pose_all), float(fps)


def find_videos(raw_dir):
    """Find videos inside folders named like '48. Hello'. Returns a list of dicts."""
    items = []
    for p in sorted(Path(raw_dir).rglob("*")):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        m = CLASS_DIR_RE.match(p.parent.name)
        if not m:
            continue
        class_id = int(m.group(1))
        class_name = m.group(2)
        slug = re.sub(r"[^a-z0-9]+", "_", class_name.lower()).strip("_")
        digits = re.findall(r"\d+", p.stem)
        items.append(
            {
                "video": p,
                "class_id": class_id,
                "class_name": class_name,
                "slug": slug,
                "file_number": int(digits[-1]) if digits else -1,
            }
        )
    return items


def stats_from(hands, pose):
    has_any = ~np.isnan(hands[:, :, 0, 0])  # (frames, 2): hand present?
    return {
        "frames": hands.shape[0],
        "hand_rate": round(float(np.mean(has_any.any(axis=1))), 3),
        "both_hands_rate": round(float(np.mean(has_any.all(axis=1))), 3),
        "pose_rate": round(float(np.mean(~np.isnan(pose[:, 0, 0]))), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw", help="folder with the extracted INCLUDE videos")
    parser.add_argument("--out", default="data/landmarks", help="where to save landmark files")
    parser.add_argument("--limit", type=int, default=0, help="only first N videos per word (testing)")
    args = parser.parse_args()

    for model in (HAND_MODEL, POSE_MODEL):
        if not model.exists():
            raise FileNotFoundError(
                f"Missing {model}. Run src/live/landmark_viewer.py once to download the models."
            )

    items = find_videos(args.raw)
    if args.limit > 0:
        counts, limited = {}, []
        for it in items:
            counts[it["class_id"]] = counts.get(it["class_id"], 0) + 1
            if counts[it["class_id"]] <= args.limit:
                limited.append(it)
        items = limited
    print(f"Found {len(items)} videos to process.")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.time()

    try:
        for n, it in enumerate(items, 1):
            class_dir = out_root / f"{it['class_id']:03d}_{it['slug']}"
            class_dir.mkdir(parents=True, exist_ok=True)
            out_file = class_dir / f"{it['video'].stem}.npz"

            if out_file.exists():
                data = np.load(out_file)
                hands, pose, status = data["hands"], data["pose"], "skipped"
            else:
                result = process_video(it["video"])
                if result is None:
                    print(f"[{n}/{len(items)}] FAILED to read {it['video']}")
                    continue
                hands, pose, fps = result
                np.savez_compressed(out_file, hands=hands, pose=pose, fps=fps)
                status = "done"

            st = stats_from(hands, pose)
            rows.append(
                {
                    "class_id": it["class_id"],
                    "class_name": it["class_name"],
                    "video_name": it["video"].stem,
                    "file_number": it["file_number"],
                    "landmark_file": str(out_file).replace("\\", "/"),
                    **st,
                }
            )
            elapsed = time.time() - start
            print(f"[{n}/{len(items)}] {status} {it['class_name']}/{it['video'].name} "
                  f"frames={st['frames']} hand_rate={st['hand_rate']} ({elapsed:.0f}s)")
    finally:
        # Save the manifest even if you stop with Ctrl+C
        if rows:
            manifest = out_root / "manifest.csv"
            with open(manifest, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote {manifest} ({len(rows)} videos)")


if __name__ == "__main__":
    main()
