---
language: es
license: cc-by-4.0
base_model: dccuchile/bert-base-spanish-wwm-cased
pipeline_tag: text-classification
tags:
  - fact-checking
  - fake-news
  - misinformation
  - colombia
  - beto
  - spanish
metrics:
  - f1
widget:
  - text: "Una persona fallecida fue jurado de votación en las elecciones de 2026"
---

# BETO — ColombiaCheck Verdict Classifier

> ⚠️ **This is NOT a truth detector.** The model predicts **how
> [ColombiaCheck](https://colombiacheck.com) would label** a claim
> (`Falso` / `Cuestionable` / `Verdadero`), learned from a single
> organization's published fact-checks. It does not verify facts, and its
> minority class **`Verdadero` scores F1 = 0.0 on the held-out test set** —
> in practice it behaves as a `Falso`/`Cuestionable` discriminator. It is an
> **educational / portfolio artifact**; do not use it for moderation,
> editorial decisions, or any decision affecting people or publications.

Fine-tuned from [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)
(Spanish BERT, cased) on 2,935 claims fact-checked by ColombiaCheck
(2018-10-26 → 2026-07-16 snapshot). Full pipeline, EDA, and governance docs
(Datasheet, Data Statement, Model Card) live in the
[GitHub repository](https://github.com/POLUX89/NLP-Fake-News-Colombia).

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="polux89/beto-colombiacheck")
clf("Una persona fallecida fue jurado de votación en las elecciones de 2026")
# [{'label': 'Falso', 'score': ...}]
```

Input: a short claim in Spanish (the corpus averages ~10 words; `max_length=64`
covers 100% of it). Output: one of `Falso`, `Cuestionable`, `Verdadero`.

## Metrics (held-out test, n = 439)

| Metric | Value |
|---|---|
| macro-F1 | **0.405** (bootstrap 95% CI **[0.371, 0.440]**) |
| `Falso` F1 | 0.806 |
| `Cuestionable` F1 | 0.410 |
| `Verdadero` F1 | **0.000** (all 14 test examples missed) |

TF-IDF + Logistic Regression baseline: 0.386 test macro-F1. Confusion matrix
(rows = truth `Cuestionable`/`Falso`/`Verdadero`): [40, 49, 0] · [61, 266, 9] ·
[5, 9, 0].

## Training

- Data: 2,935 ColombiaCheck claims (`claimReviewed` from public ClaimReview
  markup — the *neutral* claim, not the verdict-revealing headline). Stratified
  70/15/15 split, seed 42. Class distribution: `Falso` 76.5% / `Cuestionable`
  20.3% / `Verdadero` 3.2%. **The corpus itself is not redistributed** — the
  claim texts belong to ColombiaCheck; the repository ships the code to rebuild
  it.
- Loss: cross-entropy with inverse-frequency class weights (`Verdadero` ≈ 10x).
- `max_length=64`, lr 2e-5, batch 16, ≤6 epochs, early stopping on validation
  macro-F1 (patience 2), seed 42. Trained on Apple M4 (MPS, fp32).

## Limitations & bias

- **Single-source labels:** the model reproduces ColombiaCheck's editorial
  judgment and inherits its selection bias (what gets fact-checked).
- **Structural scarcity + drift of `Verdadero`:** only 93 examples, concentrated
  in the COVID era (33 in 2020 → 2 in 2026). The class is effectively
  unlearnable from this corpus, and the label distribution the model saw no
  longer exists.
- The ClaimReview-availability selection is non-random and disproportionately
  removes minority classes (details in the repo's Data Statement).

## License & attribution

- Fine-tuned weights released under **CC BY 4.0**, inheriting the base model's
  license. BETO's authors note they cannot guarantee that all of BETO's
  pre-training data is compatible with commercial use — the same caveat applies
  here (and this model is not intended for production use anyway).
- Data source credit: **ColombiaCheck** (https://colombiacheck.com).
- Base model citation:

```bibtex
@inproceedings{CaneteCFP2020,
  title={Spanish Pre-Trained BERT Model and Evaluation Data},
  author={Cañete, José and Chaperon, Gabriel and Fuentes, Rodrigo and Ho, Jou-Hui and Kang, Hojin and Pérez, Jorge},
  booktitle={PML4DC at ICLR 2020},
  year={2020}
}
```
