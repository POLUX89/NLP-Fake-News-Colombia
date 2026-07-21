# NLP · Fake News Colombia — verdict classification over ColombiaCheck

[![CI](https://github.com/POLUX89/NLP-Fake-News-Colombia/actions/workflows/ci.yml/badge.svg)](https://github.com/POLUX89/NLP-Fake-News-Colombia/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

NLP classification of the fact-check **verdict** assigned by
[ColombiaCheck](https://colombiacheck.com), from the text of the claim being
checked, together with data- and model-governance documentation: **Model Card**,
**Datasheet** and **Data Statement**.

> ⚠️ **What this project is — and is not.**
> The model learns to reproduce **how ColombiaCheck labeled** each claim, from the
> claim text. It is **not** a truth detector nor a "fake-news detector": it learns
> surface features and topics correlated with the labels of a **single**
> organization. This limitation is stated explicitly in the
> [Model Card](docs/MODEL_CARD.md) and [Data Statement](docs/DATA_STATEMENT.md).
> Treating it honestly is a central goal of this repository.

## Project phases

| Phase | Status | Deliverable |
|-------|--------|-------------|
| 0 · Environment + repo | ✅ | `pyproject.toml` (Python 3.13), structure, CI, pre-commit |
| 1 · Reconnaissance | ✅ | [`colombiacheck_recon.py`](colombiacheck_recon.py) → `recon_output/` |
| 2 · Data acquisition + EDA | ✅ | claim corpus (`data/raw/claims.csv`) + [`notebooks/01_EDA.ipynb`](notebooks/01_EDA.ipynb) |
| 3 · NLP model | ⬜ | BETO (Spanish BERT) fine-tuning, 3-class |
| 4 · Governance + demo | 🚧 | Datasheet ✅ · Data Statement / Model Card (pending) · Streamlit app |

## The corpus

Built by harvesting the **ClaimReview markup (schema.org JSON-LD)** of every
chequeo with [`src/fake_news_co/acquisition.py`](src/fake_news_co/acquisition.py).

- **4,756** unique chequeos in the archive (recon `2026-07-20`).
- **2,941** carry a `claim_reviewed` (61.8%); after de-duplicating on the claim
  text → **2,937** modeling instances. Time span `2018-10-26` → `2026-07-16`.
- **Model input = `claim_reviewed`**: the *neutral* claim as originally made
  (~10 words), **not** the article headline. The headline states the verdict
  (e.g. *"No, this image is not…"*) and would leak the label; the neutral claim
  does not. The full article body is deliberately **not** collected (it argues
  the verdict → even worse leakage, plus copyright).
- **Label = `rating`** (the ClaimReview `reviewRating`), **not** the listing
  `verdict`. The listing verdict is derived from card text by a fragile
  first-match heuristic that mislabels any headline containing the word
  *"verdadero"*; `rating` comes from structured markup and is complete. See the
  [Datasheet](docs/DATASHEET.md).

**Modeling target** (3 classes, after merging `Verdadero pero` into `Verdadero`):

| Label | n | % |
|-------|----:|----:|
| Falso | 2,247 | 76.5 % |
| Cuestionable | 597 | 20.3 % |
| Verdadero | 93 | 3.2 % |

**Central challenge:** severe class imbalance (and only ~10 words of text per
claim) → macro-F1 as the metric, and class-weighting / resampling in Phase 3.
The ClaimReview selection is non-random and disproportionately drops the minority
classes; this bias is quantified in the Datasheet and Data Statement.

## Structure

```
.
├── colombiacheck_recon.py      # Phase 1: reconnaissance scraper
├── src/fake_news_co/
│   ├── paths.py                # canonical, root-anchored paths
│   └── acquisition.py          # Phase 2: harvest claim_reviewed
├── notebooks/01_EDA.ipynb      # Phase 2: exploratory analysis
├── data/{raw,processed}/       # datasets (git-ignored)
├── models/                     # checkpoints (git-ignored)
├── app/                        # Streamlit demo (Phase 4)
├── docs/                       # MODEL_CARD · DATASHEET · DATA_STATEMENT
└── tests/
```

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"              # base deps (recon + acquisition)
pip install -e ".[eda,nlp,app,dev]"  # full stack (notebook + model + demo)
pre-commit install                   # strips notebook outputs on commit

# Phase 1 — reconnaissance
python colombiacheck_recon.py --max-pages 5   # smoke test
python colombiacheck_recon.py                 # full archive

# Phase 2 — harvest the claim corpus (resumable, ~2 h; cached)
python -m fake_news_co.acquisition --limit 5  # smoke test
python -m fake_news_co.acquisition            # full harvest
```

## Ethics & provenance

- The scraper respects `robots.txt` (`/chequeos` is allowed), identifies itself
  in the `User-Agent`, caches locally, and throttles at 1.5 s.
- The corpus is built from **public structured data (ClaimReview)**, not
  full-text scraping. Any full-text use requires prior contact with ColombiaCheck
  (`contacto@colombiacheck.com`).
- Chequeo text is **not** redistributed: `data/` is git-ignored and an
  `nbstripout` pre-commit hook strips notebook outputs so raw claim text never
  lands in the repo. Full details in the [Datasheet](docs/DATASHEET.md).

## License

MIT for the code. Chequeo texts are ColombiaCheck's property and are **not**
redistributed in this repository.
