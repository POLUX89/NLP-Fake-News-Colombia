"""Smoke tests for the recon module. Run: pytest"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "colombiacheck_recon", ROOT / "colombiacheck_recon.py"
)
recon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recon)


def test_detect_verdict_matches_multiword_before_prefix():
    # "Verdadero pero" must win over the "Verdadero" prefix.
    assert recon.detect_verdict("Verdadero pero engañoso") == "Verdadero pero"


def test_detect_verdict_falso():
    assert recon.detect_verdict("Falso Falso: la afirmación es incorrecta") == "Falso"


def test_detect_verdict_none_when_absent():
    assert recon.detect_verdict("un texto sin veredicto conocido") is None
