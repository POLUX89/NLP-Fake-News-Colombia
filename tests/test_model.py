"""Tests for fake_news_co.model (pure logic only — no training, no downloads).

The heavy deps (torch/transformers) live in the [nlp] extra, which CI does not
install: the whole module is skipped there and runs locally instead.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from fake_news_co import model as m  # noqa: E402
from fake_news_co.features import LABELS  # noqa: E402


def test_classes_cover_features_labels_alphabetically():
    assert set(m.CLASSES) == set(LABELS)
    assert list(m.CLASSES) == sorted(m.CLASSES)  # explicit, deterministic order


def test_class_weights_inverse_frequency():
    # counts [2, 1, 1], total 4, k=3 -> total/(k*count) = [4/6, 4/3, 4/3]
    w = m.compute_class_weights(np.array([0, 0, 1, 2]), n_classes=3)
    assert np.allclose(w.numpy(), [4 / 6, 4 / 3, 4 / 3])


def test_class_weights_upweight_minority():
    # 90/9/1 split -> the rarest class gets the largest weight
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    w = m.compute_class_weights(y, n_classes=3).numpy()
    assert w[2] > w[1] > w[0]


def test_load_splits_requires_known_labels(tmp_path):
    import pandas as pd

    bad = tmp_path / "dataset.csv"
    pd.DataFrame({"text": ["x"], "label": ["Inventada"], "split": ["train"]}).to_csv(
        bad, index=False
    )
    with pytest.raises(ValueError, match="unexpected labels"):
        m.load_splits(bad)
