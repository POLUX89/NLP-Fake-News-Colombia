"""
colombiacheck_recon.py  --  v0.2
================================
Reconnaissance scraper for the ColombiaCheck chequeos archive.

PURPOSE: answer the pre-proposal questions cheaply --
  * How many chequeos exist (n)?
  * What is the verdict (label) distribution?
  * What date range does the archive cover?
  * What fields does an article page reliably expose
    (date, claim, claimant, tags, ClaimReview markup)?

It does NOT download full article bodies. That is deliberate: this is a
feasibility probe, and the eventual corpus design (and a courtesy email to
contacto@colombiacheck.com) should come before any full-text harvest.

CHANGELOG v0.2 (fixes found against real data, n=3,201):
  * VERDICTS now includes "Chequeo Múltiple" (was 4.7% of cards, unlabeled).
  * Article sampling reads schema.org JSON-LD first. ClaimReview markup,
    if present, yields claim text, claimant, date and rating directly.
  * Title: JSON-LD headline -> og:title -> <title> (theme has no <h1>).
  * Date: JSON-LD -> article:published_time meta ONLY. The v0.1 free-text
    regex fallback matched the masthead's CURRENT date and silently
    corrupted every sampled row (100% filled, 100% wrong). An honest None
    beats a confident wrong value.
  * Summary now includes a date-plausibility check and ClaimReview
    coverage, and reports claim/claimant fill rates.

Verified structure (June 2026):
  * Drupal 8 site
  * Listing: https://colombiacheck.com/chequeos?page=N   (0-indexed)
  * Articles: https://colombiacheck.com/chequeos/<slug>
  * Verdict badges on cards: Falso / Cuestionable / Verdadero /
    Verdadero pero / Chequeo Múltiple (+ legacy labels pre-2018)

Politeness: identifies itself, obeys robots.txt, sleeps between requests,
caches everything locally so re-runs never re-fetch.

Usage:
  pip install requests beautifulsoup4 lxml pandas
  python colombiacheck_recon.py --max-pages 1000 --article-sample 50
  python colombiacheck_recon.py --max-pages 5            # smoke test

Outputs (./recon_output/):
  chequeos_recon.csv      one row per chequeo card found
  article_sample.csv      sampled article-level fields
  run_meta.json           run metadata
  summary printed to stdout
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://colombiacheck.com"
LISTING = BASE + "/chequeos"
SITEMAP = BASE + "/sitemap.xml"

# Identify yourself: polite scraping 101. Put your real contact here.
HEADERS = {
    "User-Agent": (
        "ColombiaCheck-recon/0.2 (academic research, Universidad EAN; "
        "contact: dsacris14723@universidadean.edu.co)"
    )
}

# Current scale (since 2018-11-23) + "Chequeo Múltiple" + legacy labels.
# Multi-word labels MUST come before their single-word prefixes.
VERDICTS = [
    "Chequeo Múltiple",
    "Verdadero pero",
    "Verdadero",
    "Cuestionable",
    "Falso",
    "Engañoso",
    "Inflado",
    "Aproximado",
    "Ligero",
]

SLEEP_SECONDS = 1.5
CACHE_DIR = Path("recon_cache")
OUT_DIR = Path("recon_output")


# ---------------------------------------------------------------- plumbing
def get(url: str, session: requests.Session) -> str | None:
    """Cached, throttled GET. Returns HTML text or None."""
    CACHE_DIR.mkdir(exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9]+", "_", url)[:150] + ".html"
    cached = CACHE_DIR / key
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    for attempt in range(3):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            time.sleep(SLEEP_SECONDS)
            if resp.status_code == 200:
                cached.write_text(resp.text, encoding="utf-8")
                return resp.text
            if resp.status_code == 404:
                return None
            print(f"  [{resp.status_code}] retry {attempt + 1} for {url}")
            time.sleep(5 * (attempt + 1))
        except requests.RequestException as exc:
            print(f"  [error] {exc} -- retry {attempt + 1}")
            time.sleep(5 * (attempt + 1))
    return None


def robots_allows(path: str) -> bool:
    rp = robotparser.RobotFileParser()
    rp.set_url(BASE + "/robots.txt")
    try:
        rp.read()
    except Exception:
        print("WARNING: could not read robots.txt; proceeding cautiously.")
        return True
    return rp.can_fetch(HEADERS["User-Agent"], BASE + path)


def detect_verdict(text: str) -> str | None:
    """Cards repeat the badge text ('Falso Falso ...'); match known labels."""
    for label in VERDICTS:
        if re.search(rf"\b{re.escape(label)}\b", text, flags=re.IGNORECASE):
            return label
    return None


# ------------------------------------------------------- JSON-LD extraction
def _walk_jsonld(obj):
    """Yield every dict found inside a JSON-LD payload (handles @graph)."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_jsonld(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_jsonld(item)


def extract_structured(soup: BeautifulSoup) -> dict:
    """Pull ClaimReview / Article fields from JSON-LD blocks if present."""
    out = {"claim_reviewed": None, "claimant": None, "rating": None,
           "date_jsonld": None, "headline": None, "keywords": None,
           "jsonld_types": []}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            payload = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        for node in _walk_jsonld(payload):
            t = node.get("@type")
            t_list = t if isinstance(t, list) else [t]
            t_list = [x for x in t_list if x]
            out["jsonld_types"] += t_list
            if "ClaimReview" in t_list:
                out["claim_reviewed"] = node.get("claimReviewed")
                rating = node.get("reviewRating") or {}
                out["rating"] = (rating.get("alternateName")
                                 or rating.get("ratingValue"))
                item = node.get("itemReviewed") or {}
                author = item.get("author") or {}
                if isinstance(author, dict):
                    out["claimant"] = author.get("name")
                out["date_jsonld"] = (node.get("datePublished")
                                      or out["date_jsonld"])
            if any(x in t_list for x in ("Article", "NewsArticle")):
                out["headline"] = out["headline"] or node.get("headline")
                out["date_jsonld"] = (out["date_jsonld"]
                                      or node.get("datePublished"))
                kw = node.get("keywords")
                if kw:
                    out["keywords"] = (", ".join(map(str, kw))
                                       if isinstance(kw, list) else str(kw))
    out["jsonld_types"] = "|".join(sorted(set(out["jsonld_types"]))) or None
    return out


# ---------------------------------------------------------------- phase A
def probe_sitemap(session: requests.Session) -> pd.DataFrame:
    """Drupal usually exposes a sitemap; gives URL + lastmod for free."""
    print("Phase A: probing sitemap.xml ...")
    xml = get(SITEMAP, session)
    rows = []
    if xml:
        soup = BeautifulSoup(xml, "xml")
        sub_sitemaps = [loc.text for loc in soup.select("sitemap > loc")]
        url_nodes = list(soup.select("url"))
        for sm in sub_sitemaps:
            child = get(sm, session)
            if child:
                url_nodes += BeautifulSoup(child, "xml").select("url")
        for node in url_nodes:
            loc = node.find("loc")
            lastmod = node.find("lastmod")
            if loc and "/chequeos/" in loc.text:
                rows.append(
                    {"url": loc.text.strip(),
                     "lastmod": lastmod.text.strip() if lastmod else None}
                )
    df = pd.DataFrame(rows).drop_duplicates(subset="url") if rows else pd.DataFrame(
        columns=["url", "lastmod"]
    )
    print(f"  sitemap chequeo URLs found: {len(df)}")
    return df


# ---------------------------------------------------------------- phase B
def crawl_listing(session: requests.Session, max_pages: int) -> pd.DataFrame:
    """Walk /chequeos?page=N extracting card-level fields."""
    print("Phase B: crawling listing pages ...")
    rows, empty_streak = [], 0
    for page in range(max_pages):
        html = get(f"{LISTING}?page={page}", session)
        if html is None:
            break
        soup = BeautifulSoup(html, "lxml")
        cards = 0
        for a in soup.find_all("a", href=re.compile(r"/chequeos/[^?#]+")):
            href = urljoin(BASE, a["href"])
            text = " ".join(a.get_text(" ", strip=True).split())
            if len(text) < 30:          # nav links, skip
                continue
            verdict = detect_verdict(text)
            body = text
            if verdict:
                body = re.sub(
                    rf"^(?:{re.escape(verdict)}\s*)+", "", body,
                    flags=re.IGNORECASE,
                ).strip()
            rows.append(
                {"url": href, "verdict": verdict, "card_text": body,
                 "listing_page": page}
            )
            cards += 1
        print(f"  page {page}: {cards} cards")
        if cards == 0:
            empty_streak += 1
            if empty_streak >= 2:       # two empty pages -> end of archive
                break
        else:
            empty_streak = 0
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset="url", keep="first")
    print(f"  unique chequeos from listing: {len(df)}")
    return df


# ---------------------------------------------------------------- phase C
def sample_articles(session: requests.Session, urls: list[str],
                    k: int) -> pd.DataFrame:
    """Fetch k articles; JSON-LD first, HTML fallbacks second."""
    print(f"Phase C: sampling {min(k, len(urls))} article pages ...")
    rows = []
    step = max(1, len(urls) // max(k, 1))      # spread across the archive
    for url in urls[::step][:k]:
        html = get(url, session)
        if not html:
            continue
        soup = BeautifulSoup(html, "lxml")
        sd = extract_structured(soup)

        # --- title: JSON-LD headline -> og:title -> <title> ---
        title = sd["headline"]
        if not title:
            og = soup.find("meta", attrs={"property": "og:title"})
            title = og["content"] if og and og.get("content") else None
        if not title and soup.title:
            title = re.sub(r"\s*\|\s*ColombiaCheck\s*$", "",
                           soup.title.get_text(strip=True))

        # --- date: JSON-LD -> article:published_time meta. NO free-text
        #     regex: the masthead shows the CURRENT date and v0.1 matched
        #     it, silently corrupting every row. ---
        date = sd["date_jsonld"]
        if not date:
            meta = soup.find("meta",
                             attrs={"property": "article:published_time"})
            date = meta["content"] if meta and meta.get("content") else None

        rows.append({
            "url": url,
            "title": title,
            "pub_date_raw": date,
            "verdict_in_page": sd["rating"]
            or detect_verdict(soup.get_text(" ", strip=True)[:3000]),
            "claim_reviewed": sd["claim_reviewed"],
            "claimant": sd["claimant"],
            "tags": sd["keywords"],
            "jsonld_types": sd["jsonld_types"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- summary
def summarize(listing: pd.DataFrame, sitemap: pd.DataFrame,
              sample: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("RECONNAISSANCE SUMMARY")
    print("=" * 60)
    n_listing, n_sitemap = len(listing), len(sitemap)
    print(f"Chequeos via listing crawl : {n_listing}")
    print(f"Chequeos via sitemap       : {n_sitemap}")

    if not listing.empty:
        print("\nVerdict distribution (listing cards):")
        dist = listing["verdict"].value_counts(dropna=False)
        for label, count in dist.items():
            pct = 100 * count / n_listing
            print(f"  {str(label):<18} {count:>6}  ({pct:.1f}%)")
        missing = listing["verdict"].isna().mean()
        if missing > 0.10:
            print(f"  NOTE: {missing:.0%} cards without detected verdict -> "
                  "inspect recon_cache HTML and extend VERDICTS/selectors.")

    if not sitemap.empty and sitemap["lastmod"].notna().any():
        lm = pd.to_datetime(sitemap["lastmod"], errors="coerce").dropna()
        if len(lm):
            print(f"\nSitemap lastmod range      : {lm.min().date()} -> "
                  f"{lm.max().date()}")

    if not sample.empty:
        print(f"\nArticle sample (n={len(sample)}):")
        for col in ("title", "pub_date_raw", "claim_reviewed",
                    "claimant", "tags"):
            if col in sample:
                print(f"  {col:<16} filled: "
                      f"{sample[col].notna().mean():.0%}")

        # ClaimReview coverage -> acquisition-strategy decision
        if sample["jsonld_types"].notna().any():
            has_cr = sample["jsonld_types"].str.contains(
                "ClaimReview", na=False).mean()
            print(f"  ClaimReview markup on {has_cr:.0%} of sampled pages")
            if has_cr > 0.5:
                print("  >>> Corpus can be built from structured data, or "
                      "via the Google Fact Check Tools API "
                      "(reviewPublisherSiteFilter=colombiacheck.com). <<<")

        # Date-plausibility check: fill rate alone hides wrong values.
        dates = pd.to_datetime(sample["pub_date_raw"], errors="coerce",
                               utc=True).dropna()
        if len(dates):
            today = pd.Timestamp.now(tz="UTC").normalize()
            same_day = (dates.dt.normalize() == today).mean()
            print(f"  date plausibility: {dates.nunique()} distinct dates, "
                  f"range {dates.min().date()} -> {dates.max().date()}")
            if same_day > 0.5 or dates.nunique() <= 2:
                print("  WARNING: sampled dates look implausible (all equal "
                      "or all = today). Inspect one recon_cache HTML before "
                      "trusting this field.")

    print("\nDecision guide:")
    print("  n >= ~2,000 with clean labels -> viable for BETO fine-tuning")
    print("  ClaimReview present           -> prefer structured/API path")
    print("  next step: email ColombiaCheck before any full-text harvest")
    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pages", type=int, default=1000)
    ap.add_argument("--article-sample", type=int, default=50)
    ap.add_argument("--skip-sitemap", action="store_true")
    args = ap.parse_args()

    if "YOUR_EMAIL_HERE" in HEADERS["User-Agent"]:
        print("Edit HEADERS first: put your real contact email in the "
              "User-Agent. Identifying yourself is non-negotiable.")
        sys.exit(1)
    if not robots_allows("/chequeos"):
        print("robots.txt disallows /chequeos for this agent. Stop here and "
              "email contacto@colombiacheck.com instead.")
        sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True)
    session = requests.Session()

    sitemap = (pd.DataFrame(columns=["url", "lastmod"])
               if args.skip_sitemap else probe_sitemap(session))
    listing = crawl_listing(session, args.max_pages)
    urls = (listing["url"].tolist() if not listing.empty
            else sitemap["url"].tolist())
    sample = sample_articles(session, urls, args.article_sample)

    listing.to_csv(OUT_DIR / "chequeos_recon.csv", index=False)
    sample.to_csv(OUT_DIR / "article_sample.csv", index=False)
    meta = {"run_at": datetime.now().isoformat(timespec="seconds"),
            "version": "0.2",
            "n_listing": len(listing), "n_sitemap": len(sitemap),
            "max_pages": args.max_pages,
            "article_sample": args.article_sample}
    (OUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2))

    summarize(listing, sitemap, sample)
    print(f"\nFiles written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()