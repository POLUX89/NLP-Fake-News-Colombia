"""Phase 2 — harvest the neutral claim (`claim_reviewed`) for the full archive.

The recon (`chequeos_recon.csv`) gives us, for all ~4,756 chequeos, the URL and
the verdict badge from the *listing card*. That card text is the editor's
headline — written already knowing the verdict — so it leaks the label and is
NOT a valid model input.

The honest input is `claim_reviewed`: the claim *as it was originally made*,
exposed by the ClaimReview schema.org markup on each article page. This module
visits every article and extracts it, reusing the proven, polite fetcher and
JSON-LD extractor from `colombiacheck_recon.py` (throttled, cached, identifies
itself). Output feeds the cleaning/split step and, ultimately, BETO (Phase 3).

Design notes:
  * RESUMABLE. Every fetched page is cached on disk (recon_cache/), and the
    output CSV is checkpointed every N articles. Re-running skips URLs already
    harvested and re-reads cached HTML, so an interrupted 2-hour run continues
    cheaply.
  * ~68% of pages carry ClaimReview markup (higher on recent articles); the
    rest yield a null claim and are kept flagged (`has_claimreview=False`) for
    the cleaning step to exclude or backfill.

Usage:
    python -m fake_news_co.acquisition --limit 5      # smoke test
    python -m fake_news_co.acquisition                # full harvest (~2h)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from fake_news_co.paths import CLAIMS_CSV, RECON_CSV, ROOT

# --- reuse the recon's polite fetcher + JSON-LD extractor -------------------
# The recon lives as a top-level script at the repo root; load it by path so
# this packaged module does not duplicate the (carefully debugged) scraping
# logic. Same mechanism the test suite uses.
_spec = importlib.util.spec_from_file_location(
    "colombiacheck_recon", ROOT / "colombiacheck_recon.py"
)
recon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recon)

OUT_CSV = CLAIMS_CSV
CHECKPOINT_EVERY = 50

OUT_COLUMNS = [
    "url",
    "verdict",          # from the listing (label)
    "claim_reviewed",   # neutral claim from ClaimReview (model input)
    "rating",           # ClaimReview reviewRating (cross-check vs. verdict)
    "pub_date",         # datePublished
    "jsonld_types",
    "has_claimreview",
]


def harvest_one(url: str, session: requests.Session) -> dict | None:
    """Fetch one article and pull the ClaimReview / Article fields."""
    html = recon.get(url, session)
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml")
    sd = recon.extract_structured(soup)

    # date: JSON-LD -> article:published_time meta (same rule as recon; no
    # free-text regex, which corrupted dates in recon v0.1).
    date = sd["date_jsonld"]
    if not date:
        meta = soup.find("meta", attrs={"property": "article:published_time"})
        date = meta["content"] if meta and meta.get("content") else None

    types = sd["jsonld_types"] or ""
    return {
        "claim_reviewed": sd["claim_reviewed"],
        "rating": sd["rating"],
        "pub_date": date,
        "jsonld_types": sd["jsonld_types"],
        "has_claimreview": "ClaimReview" in types,
    }


def load_targets(recon_csv: Path) -> pd.DataFrame:
    """Load (url, verdict) pointers produced by the recon."""
    df = pd.read_csv(recon_csv)
    return df[["url", "verdict"]].drop_duplicates(subset="url").reset_index(drop=True)


def load_done(out_csv: Path) -> pd.DataFrame:
    """Load already-harvested rows so a re-run resumes instead of restarting."""
    if out_csv.exists():
        done = pd.read_csv(out_csv)
        # keep only rows that actually resolved to something we can reuse
        return done
    return pd.DataFrame(columns=OUT_COLUMNS)


def _write(rows: list[dict], done: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset="url", keep="last")
    combined.to_csv(out_csv, index=False, columns=OUT_COLUMNS)


def harvest(limit: int | None = None,
            recon_csv: Path = RECON_CSV,
            out_csv: Path = OUT_CSV) -> pd.DataFrame:
    targets = load_targets(recon_csv)
    done = load_done(out_csv)
    done_urls = set(done["url"]) if not done.empty else set()

    todo = targets[~targets["url"].isin(done_urls)]
    if limit is not None:
        todo = todo.head(limit)

    print(f"targets: {len(targets)} | already done: {len(done_urls)} | "
          f"to harvest now: {len(todo)}")

    session = requests.Session()
    new_rows: list[dict] = []
    for i, (_, row) in enumerate(todo.iterrows(), start=1):
        fields = harvest_one(row["url"], session)
        if fields is None:
            fields = {"claim_reviewed": None, "rating": None, "pub_date": None,
                      "jsonld_types": None, "has_claimreview": False}
        new_rows.append({"url": row["url"], "verdict": row["verdict"], **fields})

        if i % 25 == 0 or i == len(todo):
            got = sum(1 for r in new_rows if r["claim_reviewed"])
            print(f"  [{i}/{len(todo)}] claim_reviewed filled: {got}/{len(new_rows)}")
        if i % CHECKPOINT_EVERY == 0:
            _write(new_rows, done, out_csv)

    _write(new_rows, done, out_csv)
    final = pd.read_csv(out_csv)
    filled = final["claim_reviewed"].notna().mean() if len(final) else 0
    print(f"\nDone. corpus rows: {len(final)} | claim_reviewed filled: "
          f"{filled:.0%} | written to {out_csv}")
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="harvest only the first N pending URLs (smoke test)")
    args = ap.parse_args()

    if not RECON_CSV.exists():
        sys.exit(f"Missing {RECON_CSV}. Run colombiacheck_recon.py first.")
    harvest(limit=args.limit)


if __name__ == "__main__":
    main()
