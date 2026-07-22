# Data Statement — ColombiaCheck Verdict Corpus

> Template based on *Data Statements for NLP* (Bender & Friedman, 2018).
> Reflects the 2026-07 corpus snapshot (2,935 modeling instances); construction
> details in the [Datasheet](DATASHEET.md).

## A. Curation Rationale
Claims selected and fact-checked by ColombiaCheck, typically statements by or
about Colombian public figures on politics and current affairs. The inclusion
criterion is **ColombiaCheck's** (what they choose to check), not a
representative sample of public discourse → inherent selection bias. A second
selection applies on top: only articles exposing ClaimReview markup enter the
modeling corpus (see Section I).

## B. Language Variety
- Spanish (`es`), **Colombian** variety (es-CO).
- Register: journalistic; claims are paraphrased public statements.

## C. Speaker / Author Demographics
- Claim texts are written by ColombiaCheck's editorial team (they are not
  verbatim transcriptions of the original speaker).
- Author demographics: not published; not inferred.

## D. Annotator Demographics
- Labels (ratings) are assigned by ColombiaCheck's professional fact-checkers
  following their public methodology. Individual demographics: not available.

## E. Speech Situation
- Context: journalistic verification of public statements.
- Time span: `2018-10-26` → `2026-07-16` (negligible coverage before 2020).
- Asynchronous, written, edited.

## F. Text Characteristics
- Very short texts: ~10 words per claim on average, max 17 (max 52 BETO
  subwords). Colombian political/institutional vocabulary; frequent proper nouns
  (`petro` is the most frequent token in **every** class).

## G. Provenance
- Source: **ClaimReview markup (schema.org JSON-LD)** on public chequeo pages,
  harvested politely (robots.txt respected, identified User-Agent, 1.5 s
  throttle, local cache). No full article bodies were collected. Claim text is
  **not redistributed** in the repository.

## H. Annotation / Labels
- ColombiaCheck's rating scale. This project does **not** re-label; it uses the
  published ratings.
- **Label = the ClaimReview `rating`**, not the listing-page verdict: the listing
  verdict comes from a fragile text heuristic and was confirmed wrong in all 6
  verdict/rating discrepancies (manual review).
- `Verdadero pero` (46 usable) is merged into `Verdadero` → 3 classes.
- 4 duplicate claim texts removed (label-consistent) to prevent split leakage;
  2 single-token URL "claims" removed.

## I. Known Limitations / Bias
- **Severe class imbalance:** `Falso` 2,245 (76.5%) · `Cuestionable` 597 (20.3%)
  · `Verdadero` 93 (3.2%) on the modeling corpus.
- **Selection is structural (MNAR), and non-neutral:** claims are usable only
  when the article exposes ClaimReview markup. Per-class retention of that
  selection: `Falso` 72.6%, `Cuestionable` 63.5%, `Verdadero` 32.3%,
  `Verdadero pero` 23.4%, `Chequeo Múltiple` 0% — i.e. the selection
  **disproportionately removes the minority classes** and raises `Falso`'s share
  from 66% to 76%.
- **Label drift over time:** ColombiaCheck barely publishes `Verdadero` ratings
  anymore (33 in 2020 → 8 in 2024 → 2 in 2026). The minority class is
  concentrated in the COVID era, so models trained on this corpus face a
  distribution that no longer exists.
- Single-organization judgment: models trained on this corpus reproduce
  ColombiaCheck's labeling policy; they do **not** measure truth. Measured
  consequence: the fine-tuned classifier scores **F1 = 0.0** for `Verdadero` on
  the test split (see [Model Card](MODEL_CARD.md)).
