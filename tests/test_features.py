"""Tests for the Phase 3 data pipeline (fake_news_co.features).

Pipeline logic is tested on a synthetic frame so CI needs no raw data
(`data/raw/claims.csv` is git-ignored). One extra test runs against the real
corpus only when it is present locally.
"""

import pandas as pd
import pytest

from fake_news_co import features
from fake_news_co.paths import CLAIMS_CSV


def _fake_raw() -> pd.DataFrame:
    """Raw-shaped claims frame with the edge cases the pipeline must handle."""
    rows = []
    for i in range(80):
        rows.append((f"u-f{i}", f"afirmacion falsa numero {i}", "Falso"))
    for i in range(40):
        rows.append((f"u-c{i}", f"afirmacion cuestionable numero {i}", "Cuestionable"))
    for i in range(30):
        rows.append((f"u-v{i}", f"afirmacion verdadera numero {i}", "Verdadero"))
    # a "Verdadero pero..." row -> must normalize + merge into Verdadero
    rows.append(("u-vp", "algo verdadero pero con matices aqui", "Verdadero pero..."))
    # a single-token URL "claim" -> must be dropped
    rows.append(("u-url", "https://facebook.com/photo.php?fbid=123", "Falso"))
    # an exact duplicate claim text -> must be de-duplicated
    rows.append(("u-dup", "afirmacion falsa numero 0", "Falso"))
    # a missing claim -> must be dropped
    rows.append(("u-na", None, "Falso"))
    # a stray label outside the 3 classes -> must be dropped
    rows.append(("u-cm", "afirmacion de chequeo multiple aqui", "Chequeo Múltiple"))
    df = pd.DataFrame(rows, columns=["url", "claim_reviewed", "rating"])
    df["pub_date"] = "2021-01-01T00:00:00-05:00"
    return df


def test_build_dataset_labels_are_three_classes():
    out = features.build_dataset(_fake_raw())
    assert set(out["label"]) <= set(features.LABELS)
    assert "Verdadero pero" not in set(out["label"])  # merged away


def test_verdadero_pero_is_merged_into_verdadero():
    out = features.build_dataset(_fake_raw())
    row = out[out["text"].str.startswith("algo verdadero pero")]
    assert len(row) == 1
    assert row["label"].iloc[0] == "Verdadero"


def test_url_claim_and_missing_and_stray_label_dropped():
    out = features.build_dataset(_fake_raw())
    assert not out["text"].str.startswith("http").any()          # URL dropped
    assert out["text"].notna().all()                             # NaN dropped
    assert "afirmacion de chequeo multiple aqui" not in set(out["text"])  # stray


def test_duplicate_claim_text_deduplicated():
    out = features.build_dataset(_fake_raw())
    assert out["text"].duplicated().sum() == 0
    assert (out["text"] == "afirmacion falsa numero 0").sum() == 1


def test_binary_label_matches_three_class():
    out = features.build_dataset(_fake_raw())
    assert set(out["label_binary"]) <= {"Falso", "no-Falso"}
    assert (out.loc[out["label"] == "Falso", "label_binary"] == "Falso").all()
    assert (out.loc[out["label"] != "Falso", "label_binary"] == "no-Falso").all()


def test_splits_are_disjoint_and_cover_everything():
    out = features.add_splits(features.build_dataset(_fake_raw()))
    assert set(out["split"]) == {"train", "val", "test"}
    counts = out["split"].value_counts()
    # majority in train, and every row assigned exactly once
    assert counts["train"] > counts["val"] + counts["test"]
    assert counts.sum() == len(out)


def test_splits_are_stratified_and_reproducible():
    base = features.build_dataset(_fake_raw())
    a = features.add_splits(base, seed=42)
    b = features.add_splits(base, seed=42)
    pd.testing.assert_series_equal(a["split"], b["split"])  # deterministic
    # every class appears in every split (stratification)
    for split in ("train", "val", "test"):
        labels = set(a.loc[a["split"] == split, "label"])
        assert labels == set(features.LABELS)


@pytest.mark.skipif(not CLAIMS_CSV.exists(), reason="raw corpus not present")
def test_real_corpus_row_count():
    out = features.build_dataset(features.load_claims())
    # matches the EDA: 2941 with a claim - 4 dup - 2 URL = 2935
    assert len(out) == 2935
    assert set(out["label"]) == set(features.LABELS)
