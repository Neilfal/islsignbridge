"""Group the INCLUDE videos by signer (person), using clothing + background colour.

Why: the dataset has no signer labels, but we need them so that train and test
contain DIFFERENT people (otherwise accuracy is inflated).

How: for each video we take the middle frame, use the saved pose landmarks to
find the torso, and measure the torso colour/texture plus a patch of background.
Then k-means groups similar videos together (one group = one signer outfit).

Run from the project root (after extract_landmarks.py):
    python src/extract/assign_signers.py            # assumes 6 signers
    python src/extract/assign_signers.py --k 6

Outputs:
    data/landmarks/signers.csv            video_name, class_id, file_number, signer
    data/contact_sheets/signers_N.jpg     thumbnails grouped by signer, for checking by eye
Always check the sheets by eye. Colour grouping is a shortcut, not a guarantee.
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}
CLASS_DIR_RE = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")
SMALL_W = 320


def kmeans(X, k, n_init=20, iters=100, seed=0):
    """Plain NumPy k-means (so we don't depend on scikit-learn for this step)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centers = [X[rng.integers(len(X))]]  # k-means++ style start
        for _ in range(1, k):
            d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
            probs = d2 / d2.sum() if d2.sum() > 0 else None
            centers.append(X[rng.choice(len(X), p=probs)])
        centers = np.array(centers)
        for _ in range(iters):
            labels = ((X[:, None, :] - centers[None]) ** 2).sum(2).argmin(1)
            new = np.array([X[labels == j].mean(0) if (labels == j).any() else centers[j]
                            for j in range(k)])
            if np.allclose(new, centers):
                break
            centers = new
        inertia = ((X - centers[labels]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, labels)
    return best[1]


def index_videos(raw_dir):
    """Map (class_id, video_name) -> path."""
    index = {}
    for p in Path(raw_dir).rglob("*"):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        m = CLASS_DIR_RE.match(p.parent.name)
        if m:
            index[(int(m.group(1)), p.stem)] = p
    return index


def read_frame(path, frame_idx):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    return cv2.resize(frame, (SMALL_W, int(h * SMALL_W / w)))


def torso_box(pose_row, w, h):
    """Box around the torso from shoulders (11, 12) and hips (23, 24)."""
    pts = pose_row[[11, 12, 23, 24], :2]
    if not np.isnan(pts).any():
        cx, cy = pts[:, 0].mean() * w, pts[:, 1].mean() * h
        shoulder_w = abs(pts[0, 0] - pts[1, 0]) * w
        torso_h = abs(pts[:2, 1].mean() - pts[2:, 1].mean()) * h
        x0, x1 = int(cx - 0.4 * shoulder_w), int(cx + 0.4 * shoulder_w)
        y0, y1 = int(cy - 0.25 * torso_h), int(cy + 0.25 * torso_h)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w), min(y1, h)
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            return x0, y0, x1, y1
    # fallback: centre of the image
    return int(0.4 * w), int(0.45 * h), int(0.6 * w), int(0.7 * h)


def features(frame, pose_row):
    h, w = frame.shape[:2]
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    x0, y0, x1, y1 = torso_box(pose_row, w, h)
    torso = lab[y0:y1, x0:x1].reshape(-1, 3)
    bg = lab[0:int(0.35 * h), 0:int(0.12 * w)].reshape(-1, 3)  # top-left wall patch
    return np.concatenate([torso.mean(0), torso.std(0)[:1], bg.mean(0)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--manifest", default="data/landmarks/manifest.csv")
    parser.add_argument("--k", type=int, default=6, help="number of signers")
    parser.add_argument("--out", default="data/contact_sheets")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    index = index_videos(args.raw)

    feats, thumbs, keep = [], [], []
    for i, row in manifest.iterrows():
        path = index.get((int(row["class_id"]), row["video_name"]))
        if path is None:
            print(f"Video not found for {row['video_name']}, skipping")
            continue
        pose = np.load(row["landmark_file"])["pose"]
        idx = pose.shape[0] // 2
        frame = read_frame(path, idx)
        if frame is None:
            print(f"Could not read frame from {path}, skipping")
            continue
        feats.append(features(frame, pose[idx]))
        thumbs.append(cv2.resize(frame, (192, 108)))
        keep.append(i)

    if not feats:
        raise SystemExit("No videos could be processed. Check --raw points at the folder with the videos.")
    feats = np.array(feats)
    rows = manifest.loc[keep].reset_index(drop=True)
    X = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    labels = kmeans(X, args.k)

    # Number signers in order of first appearance, so labels are stable and readable
    order = rows.assign(c=labels).sort_values("file_number")["c"].drop_duplicates().tolist()
    remap = {old: new for new, old in enumerate(order)}
    rows["signer"] = [remap[c] for c in labels]

    out = rows[["video_name", "class_id", "file_number", "signer"]]
    signers_csv = Path(args.manifest).parent / "signers.csv"
    out.to_csv(signers_csv, index=False)
    print(f"Wrote {signers_csv}")
    print("\nVideos per signer and word (each cell should be roughly 3-4):")
    print(pd.crosstab(rows["signer"], rows["class_id"]))

    # Thumbnails grouped by signer, labelled "S<signer> <file number> c<word id>"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    order_idx = rows.sort_values(["signer", "class_id", "file_number"]).index.tolist()
    cols, per_sheet, tw, th = 10, 100, 192, 108
    for sheet_no, start in enumerate(range(0, len(order_idx), per_sheet), 1):
        chunk = order_idx[start:start + per_sheet]
        n_rows = (len(chunk) + cols - 1) // cols
        canvas = np.zeros((n_rows * th, cols * tw, 3), dtype=np.uint8)
        for k, ridx in enumerate(chunk):
            r, c = divmod(k, cols)
            thumb = thumbs[ridx].copy()
            label = f"S{rows.loc[ridx, 'signer']} {rows.loc[ridx, 'file_number']} c{rows.loc[ridx, 'class_id']}"
            cv2.rectangle(thumb, (0, 0), (len(label) * 9 + 6, 18), (0, 0, 0), -1)
            cv2.putText(thumb, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = thumb
        f = out_dir / f"signers_{sheet_no}.jpg"
        cv2.imwrite(str(f), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"Wrote {f}")


if __name__ == "__main__":
    main()
