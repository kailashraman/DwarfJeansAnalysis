"""Pin the r_t scale-separation numbers quoted in docs/plan/stage2.md and
docs/writeup/pipeline.tex to what scripts/scale_sep95.py actually computes,
so the prose cannot silently drift from the code.

The jeffreys posteriors (results/production/<key>/jeffreys/derived.npz) are
not git-tracked, so these tests skip when they are absent.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_spec = importlib.util.spec_from_file_location(
    "scale_sep95", _SCRIPTS / "scale_sep95.py"
)
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)

PROD = _REPO / "results" / "production"

# (key, median %, 95th %) exactly as quoted in the docs.
DOC_VALUES = [
    ("crater_2", 18.16, 30.10),
    ("antlia_2", 6.48, 10.09),
    ("bootes_3", 3.12, 9.99),
    ("tucana_2", 1.10, 3.63),
    ("bootes_1", 0.94, 2.06),
    ("hydrus_1", 0.57, 1.48),
    ("carina_2", 0.43, 1.42),
    ("bootes_2", 0.28, 1.07),
    ("sextans_1", 0.69, 1.27),
    ("segue_1", 0.07, 0.21),
    ("draco_1", 0.05, 0.17),
]


def _summary(key):
    npz = PROD / key / "jeffreys" / "derived.npz"
    if not npz.exists():
        pytest.skip(f"no jeffreys/derived.npz for {key}")
    f = ss.frac_beyond_rt(npz)
    return float(np.nanmedian(f)) * 100, float(np.nanpercentile(f, 95)) * 100


@pytest.mark.parametrize("key,doc_med,doc_p95", DOC_VALUES)
def test_per_system_matches_docs(key, doc_med, doc_p95):
    med, p95 = _summary(key)
    assert med == pytest.approx(doc_med, abs=0.05), f"{key} median {med:.2f} != doc {doc_med}"
    assert p95 == pytest.approx(doc_p95, abs=0.05), f"{key} 95th {p95:.2f} != doc {doc_p95}"


def test_summary_counts_match_docs():
    """The table in stage2.md: median/95th counts under 0.5/1/2% and over 5%."""
    npzs = sorted(PROD.glob("*/jeffreys/derived.npz"))
    if len(npzs) < 39:
        pytest.skip(f"need all 39 jeffreys runs, found {len(npzs)}")
    med = np.array([np.nanmedian(ss.frac_beyond_rt(p)) for p in npzs]) * 100
    p95 = np.array([np.nanpercentile(ss.frac_beyond_rt(p), 95) for p in npzs]) * 100
    assert (int(np.sum(med < 0.5)), int(np.sum(p95 < 0.5))) == (32, 24)
    assert (int(np.sum(med < 1.0)), int(np.sum(p95 < 1.0))) == (35, 30)
    assert (int(np.sum(med < 2.0)), int(np.sum(p95 < 2.0))) == (36, 34)
    assert (int(np.sum(med > 5.0)), int(np.sum(p95 > 5.0))) == (2, 3)
