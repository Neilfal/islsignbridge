"""Baseline classifiers, evaluated two ways.

Run from the project root (after build_features.py):
    python src/models/baseline.py

Evaluation:
  * Leave-one-signer-out (LOSO): train on 5 signers, test on the 6th, repeat for
    each signer. This asks "does it work on a person it has never seen?" and is
    the honest number to report.
  * Random 5-fold split: clips from the same person end up in both train and
    test. It is shown only to demonstrate how much it overestimates accuracy.

Feature views (both built from the same normalised landmarks):
  * flat:   all 32 frames side by side (keeps the movement over time)
  * pooled: mean / std / min / max of each feature over time (a compact summary)

Models: Random Forest, SVM (RBF), Logistic Regression.
Chance level is 1 / number of words (about 11% for 9 words).

Outputs: eval/baseline_results.csv, eval/baseline_confusion_best.csv/.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = Path("data/features")
OUT = Path("eval")
SEED = 0


def make_models():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
    }


def views(X):
    n = len(X)
    return {
        "flat": X.reshape(n, -1),
        "pooled": np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1),
    }


def loso(model_factory, X, y, signers):
    pred = np.empty_like(y)
    fold_acc = {}
    for s in np.unique(signers):
        test = signers == s
        model = model_factory()
        model.fit(X[~test], y[~test])
        pred[test] = model.predict(X[test])
        fold_acc[int(s)] = accuracy_score(y[test], pred[test])
    return pred, fold_acc


def random_cv(model_factory, X, y):
    pred = np.empty_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        model = model_factory()
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    return pred


def main():
    X = np.load(FEATURES / "X.npy")
    y = np.load(FEATURES / "y.npy")
    signers = np.load(FEATURES / "signers.npy")
    with open(FEATURES / "classes.json", encoding="utf-8") as f:
        names = [v for _, v in sorted(json.load(f).items(), key=lambda kv: int(kv[0]))]
    print(f"{len(X)} clips, {len(names)} words, {len(np.unique(signers))} signers. "
          f"Chance level: {100 / len(names):.0f}%\n")

    results, best = [], None
    for view_name, V in views(X).items():
        for model_name in make_models():
            factory = lambda m=model_name: make_models()[m]
            pred, fold_acc = loso(factory, V, y, signers)
            rand_pred = random_cv(factory, V, y)
            row = {
                "view": view_name,
                "model": model_name,
                "LOSO_accuracy": round(accuracy_score(y, pred), 3),
                "LOSO_macro_F1": round(f1_score(y, pred, average="macro"), 3),
                "worst_signer_acc": round(min(fold_acc.values()), 3),
                "best_signer_acc": round(max(fold_acc.values()), 3),
                "random_split_accuracy": round(accuracy_score(y, rand_pred), 3),
            }
            results.append(row)
            print(row)
            if best is None or row["LOSO_macro_F1"] > best[0]:
                best = (row["LOSO_macro_F1"], view_name, model_name, pred)

    OUT.mkdir(exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(OUT / "baseline_results.csv", index=False)
    print("\n", df.to_string(index=False))

    _, view_name, model_name, pred = best
    print(f"\nBest by LOSO macro-F1: {model_name} on '{view_name}' features")
    print(classification_report(y, pred, target_names=names, digits=2, zero_division=0))
    cm = confusion_matrix(y, pred)
    pd.DataFrame(cm, index=names, columns=names).to_csv(OUT / "baseline_confusion_best.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{model_name} ({view_name}), leave-one-signer-out")
        fig.tight_layout()
        fig.savefig(OUT / "baseline_confusion_best.png", dpi=150)
        print(f"Saved {OUT / 'baseline_confusion_best.png'}")
    except Exception as e:  # plotting is optional
        print(f"(Could not draw the confusion matrix picture: {e})")


if __name__ == "__main__":
    main()
