"""Phase 3 — BETO verdict classifier: train, evaluate, predict.

Codifies the modeling decisions validated in
[`notebooks/02_MODELING.ipynb`](../../notebooks/02_MODELING.ipynb):

  * Input   the claim text; target = 3-class `rating` (splits frozen by
            `features.py` — this module never re-splits).
  * Model   BETO (`dccuchile/bert-base-spanish-wwm-cased`), `max_length=64`
            (the EDA measured a max of 52 subwords -> 64 covers 100%).
  * Loss    weighted cross-entropy with inverse-frequency class weights
            (`Verdadero` ~10x). The stock HF Trainer ignores class weights,
            hence the `WeightedTrainer` override.
  * Select  best epoch by validation macro-F1, early stopping (patience 2).
  * Device  MPS on Apple Silicon (fp16/bf16 disabled — unsupported on MPS).
  * Report  test evaluated ONCE with the saved model: macro-F1, per-class
            report, bootstrap 95% CI, confusion matrix.

Usage:
    python -m fake_news_co.model train
    python -m fake_news_co.model evaluate --split test

For the Streamlit demo (Phase 4):
    from fake_news_co.model import load_model, predict_texts
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from fake_news_co.features import LABELS
from fake_news_co.paths import DATA_PROCESSED, MODELS

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LENGTH = 64
SEED = 42

# Alphabetical order — matches the notebook's LabelEncoder and the saved model
# config. Kept explicit so the mapping never silently depends on data order.
CLASSES = tuple(sorted(LABELS))

DATASET_CSV = DATA_PROCESSED / "dataset.csv"
CHECKPOINT_DIR = MODELS / "beto"
FINAL_DIR = MODELS / "beto-final"


def device_name() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_splits(path: Path = DATASET_CSV) -> pd.DataFrame:
    """Load the processed dataset and attach integer ids per CLASSES order."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m fake_news_co.features` first."
        )
    df = pd.read_csv(path)
    label2id = {c: i for i, c in enumerate(CLASSES)}
    unknown = set(df["label"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"unexpected labels in dataset: {unknown}")
    df["label_id"] = df["label"].map(label2id)
    return df


def compute_class_weights(label_ids: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights (sklearn's 'balanced'): total / (k * count)."""
    counts = np.bincount(label_ids, minlength=n_classes)
    return torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float)


def train(
    epochs: int = 6,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = SEED,
    final_dir: Path = FINAL_DIR,
) -> dict:
    """Fine-tune BETO on the frozen train split; save the best model. Returns
    the validation metrics of the selected (best-epoch) model."""
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    df = load_splits()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def to_ds(split: str) -> Dataset:
        d = df[df["split"] == split]
        ds = Dataset.from_dict(
            {"text": d["text"].tolist(), "labels": d["label_id"].tolist()}
        )
        return ds.map(
            lambda b: tok(b["text"], truncation=True, max_length=MAX_LENGTH),
            batched=True,
        )

    ds_train, ds_val = to_ds("train"), to_ds("val")

    weights = compute_class_weights(
        df.query("split=='train'")["label_id"].to_numpy(), len(CLASSES)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASSES),
        id2label=dict(enumerate(CLASSES)),
        label2id={c: i for i, c in enumerate(CLASSES)},
    )

    class WeightedTrainer(Trainer):
        """Stock Trainer ignores class weights -> override the loss."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                out.logits, labels, weight=weights.to(out.logits.device)
            )
            return (loss, out) if return_outputs else loss

    def metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        return {"f1_macro": f1_score(p.label_ids, preds, average="macro")}

    steps_per_epoch = math.ceil(len(ds_train) / batch_size)
    args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # keep only the best checkpoint (~440 MB each)
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=2 * batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        warmup_steps=int(0.1 * steps_per_epoch * epochs),
        weight_decay=0.01,
        fp16=False,
        bf16=False,  # MPS does not support half precision
        report_to="none",
        logging_steps=25,
        seed=seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    val_metrics = trainer.evaluate()

    final_dir = Path(final_dir)
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    print(f"model saved -> {final_dir}")
    return val_metrics


def load_model(model_dir: Path = FINAL_DIR):
    """Load the saved fine-tuned model. Returns (tokenizer, model, device)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"{model_dir} not found. Run `python -m fake_news_co.model train` first."
        )
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    dev = device_name()
    model.to(dev).eval()
    return tok, model, dev


@torch.no_grad()
def predict_texts(
    texts: list[str],
    tok=None,
    model=None,
    device: str | None = None,
    batch_size: int = 64,
) -> tuple[list[str], np.ndarray]:
    """Predict labels for raw claim texts. Returns (labels, probabilities).

    `probabilities` has shape (n, n_classes) in the model's id order."""
    if tok is None or model is None:
        tok, model, device = load_model()
    probs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(
            texts[i : i + batch_size],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.concatenate(probs)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    labels = [id2label[i] for i in probs.argmax(axis=-1)]
    return labels, probs


def evaluate(
    split: str = "test",
    model_dir: Path = FINAL_DIR,
    n_bootstrap: int = 1000,
    seed: int = SEED,
) -> dict:
    """Evaluate the SAVED model on a frozen split (no retraining).

    Reports macro-F1, per-class F1, a bootstrap 95% CI and the confusion
    matrix; also writes metrics_<split>.json next to the model."""
    df = load_splits()
    sub = df[df["split"] == split]
    tok, model, dev = load_model(model_dir)

    # Map truth through the MODEL's label mapping (source of truth), so this
    # stays correct even if CLASSES ever changes between train and eval.
    label2id = {v: int(k) for k, v in model.config.id2label.items()}
    class_names = [model.config.id2label[i] for i in range(len(label2id))]
    y_true = sub["label"].map(label2id).to_numpy()

    pred_labels, _ = predict_texts(sub["text"].tolist(), tok, model, dev)
    y_pred = np.array([label2id[label] for label in pred_labels])

    macro = f1_score(y_true, y_pred, average="macro")
    per_class = f1_score(y_true, y_pred, average=None)

    rng = np.random.default_rng(seed)
    boot = [
        f1_score(y_true[i], y_pred[i], average="macro")
        for i in (
            rng.integers(0, len(y_true), len(y_true)) for _ in range(n_bootstrap)
        )
    ]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"--- {split} (n={len(sub)}) ---")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    print(f"macro-F1: {macro:.4f}   95% CI: [{lo:.3f}, {hi:.3f}]")

    results = {
        "split": split,
        "n": int(len(sub)),
        "macro_f1": round(float(macro), 4),
        "macro_f1_ci95": [round(float(lo), 3), round(float(hi), 3)],
        "per_class_f1": {
            c: round(float(f), 4) for c, f in zip(class_names, per_class)
        },
        "confusion_matrix": {
            "labels": class_names,
            "rows_true": confusion_matrix(y_true, y_pred).tolist(),
        },
    }
    out_path = Path(model_dir) / f"metrics_{split}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"metrics written -> {out_path}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="fine-tune BETO on the frozen train split")
    tr.add_argument("--epochs", type=int, default=6)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--learning-rate", type=float, default=2e-5)
    tr.add_argument("--seed", type=int, default=SEED)

    ev = sub.add_parser("evaluate", help="evaluate the saved model on a split")
    ev.add_argument("--split", choices=["train", "val", "test"], default="test")

    args = ap.parse_args()
    if args.cmd == "train":
        print(
            train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
            )
        )
    else:
        evaluate(split=args.split)


if __name__ == "__main__":
    main()
