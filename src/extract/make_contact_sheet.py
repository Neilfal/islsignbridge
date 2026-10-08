"""Make contact sheets (grids of thumbnails) of the INCLUDE videos.

Purpose: look at who/what is in each video, so we can tell which clips share a
signer or background, and split train/test without leaking the same person
into both sides.

Run from the project root:
    python src/extract/make_contact_sheet.py

Output: data/contact_sheets/sheet_1.jpg, sheet_2.jpg, ...
Each thumbnail is labelled  <file number> c<word id>  (e.g. "29 c48" = MVI_0029, word 48 = Hello).
Thumbnails are sorted in recording order (9914..9999 first, then 0001..0116).
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}
CLASS_DIR_RE = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")


def find_videos(raw_dir):
    items = []
    for p in Path(raw_dir).rglob("*"):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        m = CLASS_DIR_RE.match(p.parent.name)
        if not m:
            continue
        digits = re.findall(r"\d+", p.stem)
        number = int(digits[-1]) if digits else -1
        items.append((number, int(m.group(1)), p))
    # Camera counter wraps from 9999 to 0001, so put 99xx before 00xx
    items.sort(key=lambda t: (t[0] < 5000, t[0], t[1]))
    return items


def middle_frame(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/contact_sheets")
    parser.add_argument("--thumb-width", type=int, default=192)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--per-sheet", type=int, default=100)
    args = parser.parse_args()

    items = find_videos(args.raw)
    print(f"Found {len(items)} videos.")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tw = args.thumb_width
    th = int(tw * 9 / 16)

    for sheet_no, start in enumerate(range(0, len(items), args.per_sheet), 1):
        chunk = items[start:start + args.per_sheet]
        rows = (len(chunk) + args.cols - 1) // args.cols
        canvas = np.zeros((rows * th, args.cols * tw, 3), dtype=np.uint8)

        for k, (number, class_id, path) in enumerate(chunk):
            r, c = divmod(k, args.cols)
            frame = middle_frame(path)
            if frame is not None:
                thumb = cv2.resize(frame, (tw, th))
            else:
                thumb = np.zeros((th, tw, 3), dtype=np.uint8)
            label = f"{number} c{class_id}"
            cv2.rectangle(thumb, (0, 0), (len(label) * 9 + 6, 18), (0, 0, 0), -1)
            cv2.putText(thumb, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = thumb

        out_file = out_dir / f"sheet_{sheet_no}.jpg"
        cv2.imwrite(str(out_file), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
