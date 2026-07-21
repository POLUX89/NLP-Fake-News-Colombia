"""Phase 3 — data pipeline: turn the raw claim corpus into a modeling dataset.

Codifies the decisions made and justified in the EDA
([`notebooks/01_EDA.ipynb`](../../notebooks/01_EDA.ipynb)) so they become tested,
reproducible code instead of notebook state:

  1. Selection  keep rows that carry a ClaimReview claim (drop empty claims).
  2. Junk       drop single-token "claims" that are actually URLs, not text.
  3. Label      use `rating` (ClaimReview), NOT the fragile listing `verdict`.
  4. Merge      `Verdadero pero` -> `Verdadero`  =>  three classes.
  5. Dedup      drop duplicate claim texts (prevents train/test leakage).
  6. Binary     add a secondary `Falso` vs `no-Falso` target.
  7. Split      stratified 70/15/15 hold-out with a fixed seed, frozen here.

**Golden rule:** the split is computed here, once, and frozen. Any resampling
(class weights, or SMOTE on the TF-IDF baseline) happens later and only on the
`train` split — never before the split, or test leaks into training.

The processed dataset is written to `data/processed/` (git-ignored, like the raw
data): the repo ships the code to rebuild it, not ColombiaCheck's text.

Usage:
    python -m fake_news_co.features            # build data/processed/dataset.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from fake_news_co.paths import CLAIMS_CSV, DATA_PROCESSED

SEED = 42

# `rating` normalization then the 3-class merge (see EDA section 4).
RATING_NORMALIZE = {"Verdadero pero...": "Verdadero pero"}
RATING_MERGE = {"Verdadero pero": "Verdadero"}

LABELS = ["Falso", "Cuestionable", "Verdadero"]
BINARY_POSITIVE = "Falso"  # the majority / class of interest for the binary task

SPLIT_FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}


def load_claims(path: Path = CLAIMS_CSV) -> pd.DataFrame:
    """Read the raw harvested corpus (`data/raw/claims.csv`)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m fake_news_co.acquisition` first."
        )
    return pd.read_csv(path)


def build_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the EDA cleaning/labeling decisions to the raw claims frame.

    Returns a frame with columns: ``text``, ``label``, ``label_binary``,
    ``date``, ``url`` — one row per usable claim.
    """
    df = raw.copy()

    # 3/4. label = normalized + merged `rating`.
    df["label"] = df["rating"].replace(RATING_NORMALIZE).replace(RATING_MERGE)

    # 1. selection: a model needs a claim.
    df = df.dropna(subset=["claim_reviewed"])

    # 2. drop single-token claims (they are URLs, not text — EDA section 5.1).
    n_words = df["claim_reviewed"].str.split().str.len()
    df = df[n_words > 1]

    # keep only the three modeling labels (drops anything without a valid rating).
    df = df[df["label"].isin(LABELS)]

    # 5. dedup on the claim text.
    df = df.drop_duplicates(subset=["claim_reviewed"])

    # 6. secondary binary target.
    is_pos = df["label"].eq(BINARY_POSITIVE)
    df["label_binary"] = np.where(is_pos, BINARY_POSITIVE, f"no-{BINARY_POSITIVE}")

    out = pd.DataFrame(
        {
            "text": df["claim_reviewed"].to_numpy(),
            "label": df["label"].to_numpy(),
            "label_binary": df["label_binary"].to_numpy(),
            "date": df["pub_date"].to_numpy(),
            "url": df["url"].to_numpy(),
        }
    )
    return out.reset_index(drop=True)


def add_splits(
    df: pd.DataFrame,
    seed: int = SEED,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
) -> pd.DataFrame:
    """Add a ``split`` column in {train, val, test} via a stratified hold-out.

    Stratified per ``label`` and reproducible for a given ``seed``. Implemented
    with numpy (no scikit-learn) so the pipeline and its tests stay dependency-
    light and run under the CI `dev` environment.
    """
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    split = np.array(["train"] * len(df), dtype=object)

    for _, group in df.groupby("label", sort=True):
        idx = rng.permutation(group.index.to_numpy())  # shuffled copy (writable)
        n = len(idx)
        n_train = round(n * fractions["train"])
        n_val = round(n * fractions["val"])
        split[idx[n_train : n_train + n_val]] = "val"
        split[idx[n_train + n_val :]] = "test"

    out = df.copy()
    out["split"] = split
    return out


def build_processed(
    path: Path = CLAIMS_CSV,
    out_dir: Path = DATA_PROCESSED,
    seed: int = SEED,
) -> tuple[pd.DataFrame, Path]:
    """Full pipeline: load -> clean/label -> split -> write CSV. Returns df, path."""
    df = add_splits(build_dataset(load_claims(path)), seed=seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset.csv"
    df.to_csv(out_path, index=False)
    return df, out_path


def summarize(df: pd.DataFrame) -> str:
    """Human-readable label distribution overall and per split."""
    lines = [f"rows: {len(df)}", f"labels: {dict(Counter(df['label']))}"]
    for name in ("train", "val", "test"):
        sub = df[df["split"] == name]
        lines.append(f"  {name:<5} n={len(sub):>5}  {dict(Counter(sub['label']))}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    df, out_path = build_processed(seed=args.seed)
    print(summarize(df))
    print(f"written -> {out_path}")


if __name__ == "__main__":
    main()
