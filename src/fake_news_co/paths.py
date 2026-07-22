"""Canonical project paths, anchored to the repo root.

Single source of truth for filesystem locations. Because paths are derived from
this file's own location (not the current working directory), they resolve the
same from a notebook, a script, CI, or any other directory. The package is
installed editable (`pip install -e .`), so `from fake_news_co.paths import
DATA_RAW` works anywhere.

    from fake_news_co.paths import DATA_RAW
    df = pd.read_csv(DATA_RAW / "claims.csv")
"""

from __future__ import annotations

from pathlib import Path

# src/fake_news_co/paths.py -> parents[2] is the repo root.
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
RECON_OUTPUT = ROOT / "recon_output"
NOTEBOOKS = ROOT / "notebooks"
ASSETS = ROOT / "assets"

# Well-known files.
RECON_CSV = RECON_OUTPUT / "chequeos_recon.csv"
CLAIMS_CSV = DATA_RAW / "claims.csv"

__all__ = [
    "ROOT",
    "DATA",
    "DATA_RAW",
    "DATA_PROCESSED",
    "MODELS",
    "RECON_OUTPUT",
    "NOTEBOOKS",
    "ASSETS",
    "RECON_CSV",
    "CLAIMS_CSV",
]
