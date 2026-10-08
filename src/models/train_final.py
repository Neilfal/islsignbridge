"""Train the demo model on ALL clips and save it.

Run from the project root (after build_features.py):
    python src/models/train_final.py

Important: this model is for the live demo only. Its accuracy on the training
clips means nothing. The honest accuracy numbers come from
src/models/baseline.py (leave-one-signer-out).

Output: data/models/isl_rf.joblib  (ignored by Git; share it via a GitHub Release)
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

FEATURES = Path("data/features")
OUT = Path("data/models/isl_rf.joblib")


def main():
    X = np.load(FEATURES / "X.npy")
    y = np.load(FEATURES / "y.npy")
    with open(FEATURES / "classes.json", encoding="utf-8") as f:
        names = [v for _, v in sorted(json.load(f).items(), key=lambda kv: int(kv[0]))]

    model = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    model.fit(X.reshape(len(X), -1), y)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "names": names, "n_frames": X.shape[1]}, OUT, compress=3)
    size_mb = OUT.stat().st_size / 1e6
    print(f"Trained on {len(X)} clips, words: {names}")
    print(f"Saved {OUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
