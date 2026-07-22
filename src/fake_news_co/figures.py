"""Regenerate every README figure from the frozen dataset and saved artifacts.

Single reproducible entry point for the images embedded in the README, so a
future data refresh only needs: rebuild dataset -> retrain/evaluate -> rerun
this. Figures are written to `assets/` (committed; they are aggregates only —
no raw claim text beyond single tokens).

EDA figures (from `data/processed/dataset.csv`):
  * label_distribution.png          3-class target distribution
  * corpus_length_distribution.png  claim length histogram + KDE by class
  * top_terms_per_class.png         top-10 tokens per class (NLTK es pipeline)
  * tsne_word2vec.png               Word2Vec + t-SNE of top-50 tokens

Model figures (no torch needed — read from saved artifacts):
  * model_comparison.png            baseline vs BETO macro-F1 (val/test);
                                    the TF-IDF baseline is retrained here in
                                    seconds with its frozen config, BETO comes
                                    from models/beto-final/metrics_<split>.json
                                    (written by `fake_news_co.model evaluate`)
  * confusion_matrix_test.png       BETO test confusion matrix (from the JSON)

Usage:
    python -m fake_news_co.figures              # everything
    python -m fake_news_co.figures --only eda
    python -m fake_news_co.figures --only model
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this is a batch script, never a GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fake_news_co.paths import ASSETS, DATA_PROCESSED, MODELS

DATASET_CSV = DATA_PROCESSED / "dataset.csv"
FINAL_DIR = MODELS / "beto-final"
CLASS_ORDER = ["Falso", "Cuestionable", "Verdadero"]  # majority -> minority

SEED = 42
W2V_TOP_WORDS = 50

# --- Spanish text pipeline (same decisions as the EDA notebook) --------------
_ACCENTS = str.maketrans("áéíóúüÁÉÍÓÚÜ", "aeiouuAEIOUU")


def _strip_accents(text: str) -> str:
    return text.translate(_ACCENTS)


def _spanish_tokens(texts: pd.Series) -> list[list[str]]:
    """lowercase -> strip accents (ñ preserved) -> tokenize -> drop stopwords."""
    import nltk

    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.corpus import stopwords

    sw = {_strip_accents(w) for w in stopwords.words("spanish")}
    out = []
    for text in texts:
        t = re.sub(r"[^\w\s]", "", _strip_accents(str(text).lower()))
        toks = nltk.tokenize.word_tokenize(t, language="spanish")
        out.append([w for w in toks if w.isalpha() and w not in sw and len(w) > 2])
    return out


def load_dataset() -> pd.DataFrame:
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"{DATASET_CSV} not found. Run `python -m fake_news_co.features` first."
        )
    return pd.read_csv(DATASET_CSV)


def _annotate_bars(ax, fontsize=10):
    for bar in ax.patches:
        h = bar.get_height()
        ax.annotate(
            f"{int(h)}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
        )


# ------------------------------------------------------------------ EDA figs
def fig_label_distribution(df: pd.DataFrame) -> Path:
    counts = df["label"].value_counts().reindex(CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="pastel", legend=False, ax=ax)
    _annotate_bars(ax)
    ax.set_title(
        "Label distribution (3 classes, 'Verdadero pero' merged)",
        fontsize=15, fontweight="bold", pad=15,
    )
    ax.set_ylabel("Count")
    sns.despine(ax=ax)
    out = ASSETS / "label_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_corpus_length(df: pd.DataFrame) -> Path:
    d = df.copy()
    d["length"] = d["text"].str.split().str.len()
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Claim length (words)", fontsize=15, fontweight="bold")
    sns.histplot(
        data=d, x="length", hue="label", hue_order=CLASS_ORDER,
        binwidth=1, multiple="stack", ax=ax[0],
    )
    ax[0].set_title("Histogram (stacked by class)", fontsize=11, fontweight="bold")
    ax[0].set_xlabel("Words per claim")
    sns.kdeplot(
        data=d, x="length", hue="label", hue_order=CLASS_ORDER, fill=True, ax=ax[1]
    )
    ax[1].set_title("Density by class", fontsize=11, fontweight="bold")
    ax[1].set_xlabel("Words per claim")
    for a in ax:
        sns.despine(ax=a)
    out = ASSETS / "corpus_length_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_top_terms(df: pd.DataFrame, top_n: int = 10) -> Path:
    tokens = _spanish_tokens(df["text"])
    d = df.copy()
    d["tokens"] = tokens
    fig, axes = plt.subplots(1, len(CLASS_ORDER), figsize=(16, 6))
    fig.suptitle("Top-10 most frequent terms per class", fontsize=15, fontweight="bold")
    for ax, cls in zip(axes, CLASS_ORDER):
        counter = Counter(
            t for toks in d.loc[d["label"] == cls, "tokens"] for t in toks
        )
        words, counts = zip(*counter.most_common(top_n))
        sns.barplot(x=list(counts), y=list(words), color="steelblue", ax=ax)
        ax.set_title(cls, fontweight="bold")
        ax.set_xlabel("count")
        sns.despine(ax=ax)
    fig.tight_layout()
    out = ASSETS / "top_terms_per_class.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_tsne_word2vec(df: pd.DataFrame) -> Path:
    from gensim.models import Word2Vec
    from sklearn.manifold import TSNE

    tokens = _spanish_tokens(df["text"])
    # workers=1 + fixed seed -> deterministic embedding (README reproducibility)
    w2v = Word2Vec(
        sentences=tokens, vector_size=100, min_count=2,
        sg=1, seed=SEED, workers=1, epochs=30,
    )
    words = w2v.wv.index_to_key[:W2V_TOP_WORDS]
    vectors = np.asarray([w2v.wv[w] for w in words])
    xy = TSNE(
        n_components=2, random_state=SEED,
        perplexity=min(5, len(vectors) - 1),
    ).fit_transform(vectors)

    freq = Counter(t for toks in tokens for t in toks)
    colors = [freq[w] for w in words]
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=colors, cmap="viridis", s=30)
    fig.colorbar(sc, ax=ax, label="frequency")
    for i, w in enumerate(words):
        ax.text(xy[i, 0] + 0.5, xy[i, 1] + 0.5, w, fontsize=8)
    ax.set_title(
        f"t-SNE of Word2Vec embeddings ({len(words)} most frequent tokens)",
        fontsize=14, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    sns.despine(ax=ax)
    out = ASSETS / "tsne_word2vec.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- model figs
def _baseline_macro_f1(df: pd.DataFrame) -> dict[str, float]:
    """Retrain the frozen-config TF-IDF baseline (seconds, deterministic) and
    return its macro-F1 on val/test. C=1 was the notebook's grid winner."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import Pipeline

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2))),
            ("clf", LogisticRegression(
                C=1, max_iter=100_000, random_state=SEED,
                class_weight="balanced", solver="lbfgs",
            )),
        ]
    )
    tr = df[df["split"] == "train"]
    pipe.fit(tr["text"], tr["label"])
    out = {}
    for split in ("val", "test"):
        s = df[df["split"] == split]
        out[split] = f1_score(
            s["label"], pipe.predict(s["text"]), average="macro"
        )
    return out


def _beto_metrics(split: str) -> dict | None:
    path = FINAL_DIR / f"metrics_{split}.json"
    return json.loads(path.read_text()) if path.exists() else None


def fig_model_comparison(df: pd.DataFrame) -> Path | None:
    beto = {s: _beto_metrics(s) for s in ("val", "test")}
    if not all(beto.values()):
        print(
            "  ! skipping model_comparison: run "
            "`python -m fake_news_co.model evaluate --split val` and `--split test`"
        )
        return None
    base = _baseline_macro_f1(df)
    rows = pd.DataFrame(
        {
            "model": ["TF-IDF + LogReg"] * 2 + ["BETO (fine-tuned)"] * 2,
            "split": ["val", "test"] * 2,
            "macro_f1": [base["val"], base["test"],
                         beto["val"]["macro_f1"], beto["test"]["macro_f1"]],
        }
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=rows, x="model", y="macro_f1", hue="split", palette="muted", ax=ax
    )
    for bar in ax.patches:
        if bar.get_height() > 0:
            ax.annotate(
                f"{bar.get_height():.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
    ax.set_title("Macro-F1 — baseline vs BETO", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 0.55)
    sns.despine(ax=ax)
    out = ASSETS / "model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_confusion_matrix() -> Path | None:
    from sklearn.metrics import ConfusionMatrixDisplay

    metrics = _beto_metrics("test")
    if metrics is None:
        print(
            "  ! skipping confusion_matrix: run "
            "`python -m fake_news_co.model evaluate --split test`"
        )
        return None
    cm = np.asarray(metrics["confusion_matrix"]["rows_true"])
    labels = metrics["confusion_matrix"]["labels"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues", ax=ax)
    ax.set_title(
        f"BETO — test confusion matrix (n={metrics['n']})",
        fontsize=13, fontweight="bold", pad=12,
    )
    out = ASSETS / "confusion_matrix_test.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["eda", "model"], default=None)
    args = ap.parse_args()

    ASSETS.mkdir(exist_ok=True)
    df = load_dataset()
    written: list[Path | None] = []

    if args.only in (None, "eda"):
        written += [
            fig_label_distribution(df),
            fig_corpus_length(df),
            fig_top_terms(df),
            fig_tsne_word2vec(df),
        ]
    if args.only in (None, "model"):
        written += [fig_model_comparison(df), fig_confusion_matrix()]

    for path in written:
        if path is not None:
            print(f"written -> {path.relative_to(path.parents[1])}")


if __name__ == "__main__":
    main()
