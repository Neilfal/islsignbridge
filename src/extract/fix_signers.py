"""Fix the automatic signer grouping after checking it by eye.

What we saw on the contact sheets (data/contact_sheets/signers_*.jpg):
  - Signer 1 contained TWO different people:
      * the man in the dark teal shirt (file numbers 9982..9999 and 1..11), and
      * the man in the brown striped shirt, early clips (file numbers 37..47).
  - Signer 2 is the same brown-striped man, later clips (file numbers 48..63).
    The wall colour changed between his early and late clips, which is why
    colour-based grouping split him in two.

Fix: move signer 1's clips with file numbers 37..47 into signer 2.

Run from the project root:
    python src/extract/fix_signers.py

It keeps a backup of the automatic result as data/landmarks/signers_auto.csv.
"""

from pathlib import Path

import pandas as pd

CSV = Path("data/landmarks/signers.csv")
BACKUP = Path("data/landmarks/signers_auto.csv")


def main():
    df = pd.read_csv(CSV)
    if not BACKUP.exists():
        df.to_csv(BACKUP, index=False)

    early = (df["signer"] == 1) & df["file_number"].between(37, 47)
    late_ok = ((df["signer"] == 2) & df["file_number"].between(48, 63)).sum()

    if early.sum() == 0:
        print("Nothing to fix (already corrected?). Current table:")
    elif early.sum() != 11 or late_ok < 15:
        raise SystemExit(
            f"Unexpected data: {early.sum()} early clips in signer 1, {late_ok} late clips in signer 2. "
            "Your grouping differs from the one I checked, so send me the table before continuing."
        )
    else:
        df.loc[early, "signer"] = 2
        df.to_csv(CSV, index=False)
        print(f"Moved {int(early.sum())} clips from signer 1 to signer 2.")

    print("\nClips per signer and word (each cell should be 3-5):")
    print(pd.crosstab(df["signer"], df["class_id"]))
    print("\nTotal clips per signer:")
    print(df["signer"].value_counts().sort_index())


if __name__ == "__main__":
    main()
