# Datasheet — ColombiaCheck Verdict Corpus

> Template based on *Datasheets for Datasets* (Gebru et al., 2021).
> Composition and preprocessing sections reflect the Phase 2 acquisition and the
> exploratory analysis in [`notebooks/01_EDA.ipynb`](../notebooks/01_EDA.ipynb).

## Motivation
- **Why was the dataset created?** To train/evaluate an NLP classifier for the
  verdict of fact-checks, and to serve as a portfolio artifact with data- and
  model-governance documentation.
- **Who created it?** Daniel Felipe Sacristán Ávila.
- **Who funded it?** Personal project, no funding.

## Composition
- **What does each instance represent?** A claim fact-checked by ColombiaCheck,
  together with its verdict.
- **Fields (`data/raw/claims.csv`):** `url`, `verdict` (label from the listing
  card), `claim_reviewed` (the neutral claim — model input), `rating` (the
  ClaimReview `reviewRating`), `pub_date`, `has_claimreview`. `claimant` and
  `tags` were dropped (0% reliable in the JSON-LD); `jsonld_types` is dropped
  during EDA.
- **Number of instances:**
  - **Raw archive:** 4,756 unique chequeos (recon `2026-07-20`).
  - **Modeling corpus:** 2,941 rows carry a `claim_reviewed` (61.8%); after
    de-duplicating on the claim text → **2,937** instances.
- **Label used for modeling:** `rating` (from ClaimReview), **not** `verdict`
  (see *Preprocessing / Labeling*). After merging `Verdadero pero` into
  `Verdadero`, the 3-class distribution over the 2,937 modeling instances is:
  - `Falso` 2,247 (76.5%) · `Cuestionable` 597 (20.3%) · `Verdadero` 93 (3.2%).
  - Strong class imbalance; the two "true-ish" classes are tiny.
- **Raw verdict distribution** (listing labels over the 4,756 archive, for
  reference): `Falso` 65.0% (3,093), `Cuestionable` 19.8% (940),
  `Chequeo Múltiple` 5.7% (273), `Verdadero pero` 4.1% (197), `Verdadero`
  3.4% (161), unlabeled 1.9% (92).
- **Missing data?** `claim_reviewed`, `rating` and `pub_date` are missing
  *together* and *exactly* when `has_claimreview == False` (38.2% of the
  archive). This is **structural (MNAR)**: those articles carry no ClaimReview
  markup (older articles and the `Chequeo Múltiple` format), not a data-quality
  defect. `claimant`/`tags`: 0%.
- **Sensitive / personal information?** The claim text mentions public figures.
  No private personal data.

## Collection Process
- **How was it acquired?** ClaimReview markup (schema.org JSON-LD) was extracted
  from each public chequeo article. The recon feasibility probe is
  [`colombiacheck_recon.py`](../colombiacheck_recon.py); the full harvest of
  `claim_reviewed` for all 4,756 URLs is
  [`src/fake_news_co/acquisition.py`](../src/fake_news_co/acquisition.py)
  (resumable, cached, checkpointed). Structured data was preferred over
  full-text scraping.
- **Politeness:** respects `robots.txt` (`/chequeos` is allowed), identifies
  itself in the `User-Agent`, caches locally, throttles at 1.5 s.
- **Time span:** `2018-10-26` → `2026-07-16`. Coverage is negligible early
  (6 chequeos in 2018, 3 in 2019) and concentrated from 2020 onward.

## Preprocessing / Cleaning / Labeling
Decisions made and justified in the EDA:
- **Selection to the ClaimReview-covered subset** (drop rows with empty
  `claim_reviewed`): required (a model cannot learn from an empty claim) but
  **non-random**. It disproportionately removes the minority classes — per-class
  retention: `Falso` 72.6%, `Cuestionable` 63.5%, `Verdadero` 32.3%,
  `Verdadero pero` 23.4%, `Chequeo Múltiple` 0%. `Falso`'s share rises from
  66% to 76%. This selection bias is recorded in the Data Statement.
- **De-duplication on the claim text** (`claim_reviewed`), not just `url`:
  removes 4 duplicate claims (verified label-consistent) to prevent train/test
  leakage. 2,941 → 2,937.
- **Label source = `rating` (ClaimReview), not `verdict` (listing).** `verdict`
  is derived from listing-card text by matching the first known label in a fixed
  order (`Verdadero` before `Falso`), so headlines containing the word
  *"verdadero"* are mislabeled. Manual review of the 6 verdict/rating
  discrepancies confirmed `rating` correct in all 6. `rating` has 0 nulls on the
  modeling corpus.
- **`Chequeo Múltiple`** is a multi-claim format with no single ClaimReview
  claim; it has 0 usable rows and drops out naturally.
- **`Verdadero pero` merged into `Verdadero`** to yield a 3-class target and add
  support to the smallest class.

## Uses
- Verdict classification; selection-bias analysis.
- **Not recommended for:** asserting objective truth; generalizing to other
  outlets or to Spanish outside the Colombian fact-checking domain.

## Distribution & Maintenance
- **Is the corpus redistributed?** **No.** The chequeo text is ColombiaCheck's
  property and is not redistributed; the repo ships the **code** to reconstruct
  it. Notebook outputs are stripped before commit (`nbstripout` pre-commit hook)
  so raw claim text never lands in the repo.
- **Source contact:** `contacto@colombiacheck.com` (to be contacted before any
  full-text harvest).
- **Maintainer:** the author.
