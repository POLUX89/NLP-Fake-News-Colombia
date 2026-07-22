# Model Card — ColombiaCheck Verdict Classifier (BETO)

> Template based on *Model Cards for Model Reporting* (Mitchell et al., 2019).
> Metrics reflect the model trained on the 2026-07 corpus snapshot; experiments in
> [`notebooks/02_MODELING.ipynb`](../notebooks/02_MODELING.ipynb).

## Model Details
- **Developed by:** Daniel Felipe Sacristán Ávila (personal portfolio).
- **Date:** 2026-07-21.
- **Version:** 0.1.0.
- **Type:** multi-class, single-label text classifier; fine-tuned **BETO**
  (`dccuchile/bert-base-spanish-wwm-cased`).
- **Input:** the fact-checked claim text (`claim_reviewed`), Spanish, ~10 words.
- **Output:** ColombiaCheck rating ∈ {`Falso`, `Cuestionable`, `Verdadero`}
  (`Verdadero pero` merged into `Verdadero`; see [Datasheet](DATASHEET.md)).
- **License:** MIT (code). Model weights are **not** distributed in this repo
  (`models/` is git-ignored); a Hugging Face Hub release is planned for the demo.

## Intended Use
- **Intended:** educational / portfolio demonstration of a Spanish NLP pipeline,
  and an analysis of how a model reproduces one organization's fact-check labels.
- **Intended users:** technical recruiters, NLP students, the author.
- **Out of scope (do NOT use for):**
  - Determining whether a claim is objectively true or false.
  - Content moderation or automated editorial/journalistic decisions.
  - Any decision affecting people, outlets, or publications.

## Factors
- Domain: Colombian politics and current affairs; public figures' statements.
- Performance varies drastically **by class** (see Metrics) and the label
  distribution **drifts over time**: `Verdadero` fell from 33 chequeos in 2020 to
  2 in 2026, so the minority class is concentrated in the COVID era.

## Training Details
- Data: 2,935 claims, stratified 70/15/15 split frozen in
  [`features.py`](../src/fake_news_co/features.py) (seed 42). Train n=2,055.
- Loss: cross-entropy with **inverse-frequency class weights** (`Verdadero` ≈ 10x).
- Hyperparameters: `max_length=64` (covers 100% of claims), lr 2e-5, batch 16,
  ≤6 epochs with early stopping (patience 2) on validation macro-F1, weight
  decay 0.01, warmup 10%, seed 42.
- Hardware: Apple M4 (MPS), fp32 (MPS does not support fp16).
- Baseline for comparison: TF-IDF (1-2 grams, min_df=2) + Logistic Regression
  (`class_weight="balanced"`, C tuned by 3-fold CV on train). A SMOTE variant of
  the baseline was tried and was statistically equivalent (0.406 vs 0.398 val).

## Metrics
Primary metric: **macro-F1** (severe class imbalance — accuracy would be
misleading). Test metrics computed once, on the frozen test split (n=439).

| Model | macro-F1 (val) | macro-F1 (test) | Falso F1 | Cuestionable F1 | Verdadero F1 |
|---|---|---|---|---|---|
| TF-IDF + LogReg (class_weight) | 0.398 | 0.386 | 0.743 | 0.351 | 0.065 |
| **BETO (weighted loss)** | **0.449** | **0.405** | **0.806** | **0.410** | **0.000** |

- **Bootstrap 95% CI** (BETO test macro-F1, 1,000 resamples): **[0.371, 0.440]**.
- Test confusion matrix (rows = truth): `Cuestionable` [40, 49, 0] ·
  `Falso` [61, 266, 9] · `Verdadero` [5, 9, 0].
- **`Verdadero` fails completely on test (F1 = 0.0)**: all 14 test examples are
  misclassified despite the 10x class weight. Only 65 training examples exist and
  they are topically/temporally concentrated. This is a structural data
  limitation, documented rather than hidden.

## Evaluation Data
- Same corpus and split as training (val n=441, test n=439); `Verdadero` has only
  **14 test examples**, hence the bootstrap CI. See the
  [Datasheet](DATASHEET.md) for corpus construction and selection bias.

## Ethical Considerations
- The model **does not verify facts**; it reproduces the judgment of a single
  fact-checking organization and inherits its selection bias (what gets checked)
  and labeling policy.
- **Misuse risk:** presenting it as a "truth detector" or "fake-news detector".
  Mitigations: this card, the README warning, and the
  [Data Statement](DATA_STATEMENT.md).
- In practice the model behaves as a `Falso`/`Cuestionable` discriminator and
  essentially never predicts `Verdadero` — a deployment would systematically label
  true claims as false/questionable. This is a further reason it must not be used
  for real decisions.

## Caveats & Recommendations
- Corpus of moderate size (2,935) from a **single source**; results do not
  generalize beyond ColombiaCheck's editorial pipeline.
- Label drift over time (`Verdadero` nearly extinct after 2021) means performance
  on future data is likely worse than test metrics suggest.
- Recommended as a didactic artifact only. For a usable `Verdadero` class, the
  corpus would need substantially more positive examples (e.g., pooling multiple
  fact-checkers or data augmentation — both out of scope here).
