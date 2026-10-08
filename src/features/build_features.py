"""Turn the raw landmark files into fixed-size feature arrays for training.

Run from the project root (after extract_landmarks.py and fix_signers.py):
    python src/features/build_features.py

What it does, for every clip:
  1. Makes positions body-relative: x/y are measured from the middle of the
     shoulders and divided by shoulder width, so it no longer matters where
     the person stands or how far they are from the camera.
  2. Adds hand-shape features: each hand's landmarks relative to its own wrist,
     divided by hand size, so finger shapes are comparable between people.
  3. Fills gaps where a hand was not detected (linear interpolation in time),
     and adds a "hand present" flag so the model still knows it was a gap.
  4. Resamples the clip to exactly 32 frames, so a slow signer and a fast
     signer give arrays of the same shape.

Per frame (184 numbers):
    84  hand positions relative to the body (2 hands x 21 points x x,y)
    14  upper-body pose positions (nose, shoulders, elbows, wrists)
    84  hand shape relative to each hand's own wrist
     2  hand-present flags

Output (in data/features/, ignored by Git):
    X.npy        (clips, 32, 184)
    y.npy        word index 0..8
    signers.npy  signer id per clip
    meta.csv     video_name, class_id, class_name, signer, file_number, label
    classes.json label index -> word name
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

N_FRAMES = 32
# INCLUDE videos are 16:9. MediaPipe gives x as a fraction of width and y as a
# fraction of height, so x must be multiplied by width/height to get equal units.
# For the live webcam, pass the real frame width/height instead.
ASPECT = 16 / 9
POSE_IDS = [0, 11, 12, 13, 14, 15, 16]  # nose, shoulders, elbows, wrists
MANIFEST = Path("data/landmarks/manifest.csv")
SIGNERS = Path("data/landmarks/signers.csv")
OUT_DIR = Path("data/features")


def nanmedian(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def interp_columns(A):
    """Fill NaNs in each column by linear interpolation over time (all-NaN -> 0)."""
    T = A.shape[0]
    t = np.arange(T)
    out = np.zeros_like(A)
    for d in range(A.shape[1]):
        col = A[:, d]
        ok = ~np.isnan(col)
        if ok.any():
            out[:, d] = np.interp(t, t[ok], col[ok])
    return out


def resample(A, n=N_FRAMES):
    """Resample (T, D) to (n, D) by linear interpolation along time."""
    T = A.shape[0]
    if T == 1:
        return np.repeat(A, n, axis=0)
    t_old = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, n)
    out = np.empty((n, A.shape[1]), dtype=np.float32)
    for d in range(A.shape[1]):
        out[:, d] = np.interp(t_new, t_old, A[:, d])
    return out


def clip_features(hands, pose, aspect=ASPECT):
    """hands: (T,2,21,3), pose: (T,33,4) with NaN for missing. Returns (T,184)."""
    hands = hands.astype(np.float64).copy()
    pose = pose.astype(np.float64).copy()
    hands[..., 0] *= aspect
    pose[..., 0] *= aspect

    shoulders = pose[:, [11, 12], :2]                  # (T, 2, 2)
    mid = nanmedian(shoulders.mean(axis=1), axis=0)    # (2,) one reference per clip
    width = nanmedian(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1))
    if np.isnan(mid).any() or np.isnan(width) or width < 1e-3:
        mid = np.array([0.5 * aspect, 0.4])            # fallback: assume centred
        width = 0.25 * aspect

    hands_xy = hands[..., :2]                          # (T, 2, 21, 2)
    body_hands = ((hands_xy - mid) / width).reshape(len(hands), -1)
    body_pose = ((pose[:, POSE_IDS, :2] - mid) / width).reshape(len(pose), -1)

    wrist = hands_xy[:, :, 0:1, :]
    size = np.linalg.norm(hands_xy[:, :, 9, :] - hands_xy[:, :, 0, :], axis=-1)  # wrist -> middle knuckle
    size = np.where(size < 1e-6, np.nan, size)
    shape = ((hands_xy - wrist) / size[..., None, None]).reshape(len(hands), -1)

    present = (~np.isnan(hands_xy[:, :, 0, 0])).astype(np.float64)  # (T, 2)

    feats = np.concatenate([body_hands, body_pose, shape], axis=1)
    feats = interp_columns(feats)
    return np.concatenate([feats, present], axis=1)


def main():
    manifest = pd.read_csv(MANIFEST)
    signers = pd.read_csv(SIGNERS)[["video_name", "class_id", "signer"]]
    df = manifest.merge(signers, on=["video_name", "class_id"], how="inner")
    if len(df) != len(manifest):
        print(f"Warning: {len(manifest) - len(df)} clips have no signer label and are skipped.")

    class_ids = sorted(df["class_id"].unique())
    label_of = {cid: i for i, cid in enumerate(class_ids)}
    names = df.drop_duplicates("class_id").set_index("class_id")["class_name"]

    X, rows = [], []
    for _, r in df.iterrows():
        data = np.load(r["landmark_file"])
        X.append(resample(clip_features(data["hands"], data["pose"])))
        rows.append(r)

    X = np.stack(X).astype(np.float32)
    meta = pd.DataFrame(rows).reset_index(drop=True)
    meta["label"] = meta["class_id"].map(label_of)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "X.npy", X)
    np.save(OUT_DIR / "y.npy", meta["label"].to_numpy())
    np.save(OUT_DIR / "signers.npy", meta["signer"].to_numpy())
    meta[["video_name", "class_id", "class_name", "signer", "file_number", "label"]].to_csv(
        OUT_DIR / "meta.csv", index=False
    )
    with open(OUT_DIR / "classes.json", "w", encoding="utf-8") as f:
        json.dump({label_of[c]: names[c] for c in class_ids}, f, indent=2)

    print(f"X shape: {X.shape}  (clips, frames, features)")
    print(f"Words: {len(class_ids)}   Signers: {meta['signer'].nunique()}")
    print(f"Any NaN left: {bool(np.isnan(X).any())}")
    print(f"Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
