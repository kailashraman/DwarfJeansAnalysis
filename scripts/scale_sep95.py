#!/usr/bin/env python
"""Tracer/tidal-radius scale-separation audit for the Stage 2 Jeans fit.

Stage 2 applies no tidal-radius truncation to the sigma_los integrals (the
upper limit is s_inf = 100*max(r_s, r_p), deep in the Plummer r^-5 tail). This
is justified when almost none of the fitted tracer population lives beyond the
tidal radius r_t, where the single-component equilibrium Jeans model breaks
down. This script quantifies that per dwarf.

The tracer is a Plummer profile, so its 3D enclosed-mass fraction beyond r_t
has the closed form

    f = 1 - r_t^3 / (r_t^2 + r_p^2)^(3/2)

with r_p the Plummer scale radius (= circularized projected half-light radius,
stored per draw as ``r_p_chain``) and r_t the tidal radius (real-MW host,
Tormen/Springel; stored per draw as ``r_t_kpc`` on the ``idx_jd`` subsample).

f is evaluated per posterior draw (pairing ``r_p_chain[idx_jd]`` with
``r_t_kpc``) and summarized by its median and 95th percentile. f is steeply
nonlinear in r_t, so the median understates the upper tail; the 95th percentile
is the honest tail summary. Because r_t depends on the inferred mass, f is
prior-dependent -- select the posterior with ``--prior`` (default ``jeffreys``).

Reproduces the scale-separation table in docs/plan/stage2.md and the
corresponding paragraph in docs/writeup/pipeline.tex.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROD = REPO / "results" / "production"


def frac_beyond_rt(derived_npz: Path) -> np.ndarray:
    """Per-draw Plummer tracer-mass fraction beyond r_t."""
    d = np.load(derived_npz, allow_pickle=True)
    r_t = d["r_t_kpc"]                    # (n_jd,) tidal radius per draw
    r_p = d["r_p_chain"][d["idx_jd"]]     # Plummer scale on the same draws
    return 1.0 - r_t**3 / (r_t**2 + r_p**2) ** 1.5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prior", default="jeffreys",
                    help="posterior sub-run under results/production/<key>/ (default: jeffreys)")
    ap.add_argument("keys", nargs="*", help="lvdb keys (default: all with a <prior> run)")
    args = ap.parse_args()

    if args.keys:
        keys = args.keys
    else:
        keys = sorted(p.name for p in PROD.iterdir()
                      if (p / args.prior / "derived.npz").exists())

    rows = []
    for key in keys:
        npz = PROD / key / args.prior / "derived.npz"
        if not npz.exists():
            print(f"  [skip] {key}: no {args.prior}/derived.npz")
            continue
        try:
            f = frac_beyond_rt(npz)
        except KeyError as exc:  # older/partial derived.npz missing a needed key
            print(f"  [skip] {key}: missing key {exc}")
            continue
        # r_t can be filled NaN for draws where the tidal-radius solve failed
        # (see postprocess.py); nan-aware summaries so a few bad draws don't
        # poison the whole dwarf, and surface the failure fraction.
        n_nan = int(np.sum(~np.isfinite(f)))
        if n_nan:
            print(f"  [note] {key}: {n_nan}/{f.size} draws NaN (r_t solve failed)")
        rows.append((key, float(np.nanmedian(f)) * 100,
                     float(np.nanpercentile(f, 95)) * 100))

    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"Tracer-mass fraction beyond r_t  (prior='{args.prior}', N={len(rows)})")
    print(f"{'dwarf':<20} {'median %':>9} {'95th %':>9}")
    print("-" * 40)
    for key, med, p95 in rows:
        print(f"{key:<20} {med:>9.2f} {p95:>9.2f}")

    med = np.array([r[1] for r in rows])
    p95 = np.array([r[2] for r in rows])
    n = len(rows)
    print("\nCounts (out of %d):" % n)
    print(f"{'threshold':<12} {'median':>8} {'95th pct':>9}")
    for thr in (0.5, 1.0, 2.0):
        print(f"  <{thr:<9} {int(np.sum(med < thr)):>8} {int(np.sum(p95 < thr)):>9}")
    print(f"  >{5.0:<9} {int(np.sum(med > 5)):>8} {int(np.sum(p95 > 5)):>9}")


if __name__ == "__main__":
    main()
