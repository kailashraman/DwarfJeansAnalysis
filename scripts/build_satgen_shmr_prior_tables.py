"""Derive per-dwarf SHMR-weighted SatGen (r_s, ρ_s) prior tables.

For each galaxy in the 39-dwarf study sample, the Diemer halo catalog
is reweighted by the precomputed Fattahi+18 (or other) stellar-to-halo
mass relation weights from the SatGen_Dwarf sibling repo. Those
weights already include a geometric prior that zeroes any halo whose
galactocentric distance falls outside a factor of 2 of the observed
GC distance — so the effective sub-population per dwarf is much
smaller than the 2.43M halos in the raw catalog, and zero-weight
halos are dropped before any binning.

The output schema matches scripts/build_satgen_prior_table.py so the
inference loader reuses the same code path.

Outputs (one per dwarf):
  data/satgen_prior/<shmr>/<lvdb_key>.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import h5py
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
SATGEN_REPO = Path("/global/scratch/projects/pc_heptheory/kraman/SatGen_Dwarf")
SATGEN_H5 = SATGEN_REPO / "data/additional/m12res8_10k_Diemer+scatter_sim.h5"
WEIGHT_DIR = SATGEN_REPO / "data/additional/weights_gc/Diemer"
STUDY_YAML = REPO / "src/dwarfjeans/ingest/config/study_sample.yaml"
OUT_BASE = REPO / "data/satgen_prior"

# Order in SatGen_Dwarf/python/compute_weights.py::SHMR_names.
SHMR_INDEX = {
    "behroozi13": 0, "moster13": 1, "rodriguezpuebla17": 2, "fattahi18": 3,
    "moster18": 4, "behroozi19": 5, "munshi21": 6, "danieli23_const": 7,
    "danieli23_grow": 8, "kim24": 9,
}

LOG10_RS_BOUNDS = (-2.0, 1.0)
N_BINS_DEFAULT = 300
N_CDF = 1024
ESS_FLOOR_PER_BIN = 30  # require ESS / n_bins >= this; otherwise adapt n_bins.

# Derived from x_max ≈ 2.16258 solving the NFW v_c maximum (matches
# scripts/build_satgen_prior_table.py).
RS_OVER_RMAX = 0.46241029979236
RHOS_COEFF = 1.7212585601570


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_halos():
    """Return (log10_rs, log10_rhos, in_range_mask) for the full Diemer catalog.

    in_range_mask is an alignment-preserving mask onto the original
    catalog rows that is True iff (r_s, ρ_s) are finite, positive, and
    log10 r_s lies within LOG10_RS_BOUNDS. The returned log10_rs and
    log10_rhos are the masked sub-arrays.
    """
    with h5py.File(SATGEN_H5, "r") as fh:
        v_max_kpcGyr = fh["v_max"][()]
        r_max_kpc = fh["r_max"][()]
    v_max = (v_max_kpcGyr * u.kpc / u.Gyr).to(u.km / u.s).value
    G_val = const.G.to(u.km**2 * u.kpc / (u.s**2 * u.Msun)).value
    r_s = RS_OVER_RMAX * r_max_kpc
    rho_s = RHOS_COEFF * (v_max / r_max_kpc) ** 2 / G_val
    log10_rs_full = np.full(r_s.shape, np.nan)
    log10_rhos_full = np.full(rho_s.shape, np.nan)
    finite = np.isfinite(r_s) & np.isfinite(rho_s) & (r_s > 0) & (rho_s > 0)
    log10_rs_full[finite] = np.log10(r_s[finite])
    log10_rhos_full[finite] = np.log10(rho_s[finite])
    keep = (
        finite
        & (log10_rs_full >= LOG10_RS_BOUNDS[0])
        & (log10_rs_full <= LOG10_RS_BOUNDS[1])
    )
    return log10_rs_full, log10_rhos_full, keep, G_val


def _weighted_quantile_bin_edges(x_sorted, w_sorted, n_bins):
    """Bin edges at uniform quantiles of cumulative weight (midpoint).

    x_sorted, w_sorted: arrays sorted by x, w > 0.
    """
    cumw = np.cumsum(w_sorted)
    cumw_norm = (cumw - 0.5 * w_sorted) / cumw[-1]
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.interp(quantiles, cumw_norm, x_sorted)
    edges[0] = np.nextafter(x_sorted[0], -np.inf)
    edges[-1] = np.nextafter(x_sorted[-1], +np.inf)
    return edges


def _weighted_mean_std(y, w):
    """Reliability-weighted mean and Bessel-corrected std.

    σ² = Σwᵢ(yᵢ-μ)² / (Σw − Σw²/Σw)
       = (Σw)² · biased_var / ((Σw)² − Σw²).
    """
    sumw = np.sum(w)
    mu = float(np.sum(w * y) / sumw)
    sumw2 = float(np.sum(w * w))
    denom = sumw - sumw2 / sumw
    if denom <= 0.0:
        return mu, float("nan")
    var = float(np.sum(w * (y - mu) ** 2) / denom)
    return mu, np.sqrt(max(var, 0.0))


def _build_for_dwarf(study_name, lvdb_key, shmr,
                     log10_rs_full, log10_rhos_full, in_range_mask):
    weight_path = WEIGHT_DIR / f"{study_name}.npz"
    with np.load(weight_path) as wnpz:
        w_all = wnpz["mstar_weights"][SHMR_INDEX[shmr]].astype(np.float64)

    if w_all.shape[0] != log10_rs_full.shape[0]:
        raise RuntimeError(
            f"{study_name}: weight length {w_all.shape[0]} does not match "
            f"halo catalog length {log10_rs_full.shape[0]}"
        )

    keep = in_range_mask & (w_all > 0.0)
    n_total = int(in_range_mask.sum())
    n_nonzero = int(keep.sum())
    zero_frac = 1.0 - n_nonzero / max(n_total, 1)

    x = log10_rs_full[keep]
    y = log10_rhos_full[keep]
    w = w_all[keep]

    sumw = float(w.sum())
    ess = float(sumw * sumw / float((w * w).sum()))

    # Adaptive n_bins so each bin holds at least ESS_FLOOR_PER_BIN effective
    # halos. Per-bin ESS is itself reduced when one bin has highly disparate
    # weights, but ESS/n_bins is the cheap, monotone proxy used here.
    n_bins = N_BINS_DEFAULT
    if ess / n_bins < ESS_FLOOR_PER_BIN:
        n_bins = max(50, int(ess / ESS_FLOOR_PER_BIN))
    if n_bins < 50:
        raise RuntimeError(
            f"{study_name}: ESS={ess:.0f} too small for >=50 bins at floor "
            f"{ESS_FLOOR_PER_BIN}/bin; reconsider the weight construction."
        )

    order = np.argsort(x, kind="stable")
    x_sorted = x[order]
    y_sorted = y[order]
    w_sorted = w[order]

    bin_edges = _weighted_quantile_bin_edges(x_sorted, w_sorted, n_bins)

    bin_idx = np.clip(np.digitize(x, bin_edges) - 1, 0, n_bins - 1)
    bin_centers = np.zeros(n_bins)
    mu = np.full(n_bins, np.nan)
    sigma = np.full(n_bins, np.nan)
    n_per_bin = np.zeros(n_bins, dtype=int)
    ess_per_bin = np.zeros(n_bins)
    for k in range(n_bins):
        sel = bin_idx == k
        n_per_bin[k] = int(sel.sum())
        if n_per_bin[k] >= 2:
            xk = x[sel]
            yk = y[sel]
            wk = w[sel]
            sk = float(wk.sum())
            ess_per_bin[k] = sk * sk / float((wk * wk).sum())
            bin_centers[k] = float(np.sum(wk * xk) / sk)
            mu_k, sigma_k = _weighted_mean_std(yk, wk)
            mu[k] = mu_k
            sigma[k] = sigma_k

    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        bad = int((~np.isfinite(mu) | ~np.isfinite(sigma)).sum())
        raise RuntimeError(
            f"{study_name}: {bad} bin(s) had non-finite weighted moments; "
            f"n_bins={n_bins}, ess={ess:.0f}"
        )

    # Marginal CDF of log10 r_s on a fine grid, weighted.
    cdf_y = np.cumsum(w_sorted) / sumw
    log10_rs_grid = np.linspace(LOG10_RS_BOUNDS[0], LOG10_RS_BOUNDS[1], N_CDF)
    cdf_log10_rs = np.interp(log10_rs_grid, x_sorted, cdf_y, left=0.0, right=1.0)
    cdf_log10_rs = np.maximum.accumulate(cdf_log10_rs)
    cdf_log10_rs[0] = 0.0
    cdf_log10_rs[-1] = 1.0

    metadata = {
        "study_name": study_name,
        "lvdb_key": lvdb_key,
        "shmr": shmr,
        "shmr_index": SHMR_INDEX[shmr],
        "source_h5": str(SATGEN_H5),
        "source_h5_sha256": _sha256(SATGEN_H5),
        "weights_file": str(weight_path),
        "n_total_in_range": n_total,
        "n_halos_nonzero": n_nonzero,
        "zero_weight_fraction": zero_frac,
        "sum_weights": sumw,
        "ess": ess,
        "ess_per_bin_min": float(ess_per_bin.min()),
        "ess_per_bin_median": float(np.median(ess_per_bin)),
        "n_bins": int(n_bins),
        "n_bins_default": N_BINS_DEFAULT,
        "ess_floor_per_bin": ESS_FLOOR_PER_BIN,
        "n_cdf": N_CDF,
        "log10_rs_bounds": list(LOG10_RS_BOUNDS),
        "log10_rs_min_data": float(x.min()),
        "log10_rs_max_data": float(x.max()),
        "binning": "weighted_quantile",
        "variance_correction": "reliability_bessel",
    }
    return dict(
        log10_rs_grid=log10_rs_grid,
        cdf_log10_rs=cdf_log10_rs,
        bin_centers_log10_rs=bin_centers,
        bin_edges_log10_rs=bin_edges,
        mu_log10_rhos=mu,
        sigma_log10_rhos=sigma,
        n_per_bin=n_per_bin,
        ess_per_bin=ess_per_bin,
        metadata=json.dumps(metadata),
    ), metadata


def _load_study_sample():
    with open(STUDY_YAML) as f:
        y = yaml.safe_load(f)
    return [(g["study_name"], g["lvdb_key"]) for g in y["galaxies"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shmr", default="fattahi18",
                        choices=tuple(SHMR_INDEX.keys()))
    parser.add_argument("--only", default=None,
                        help="comma-separated lvdb_keys to limit the build")
    args = parser.parse_args()

    out_dir = OUT_BASE / args.shmr
    out_dir.mkdir(parents=True, exist_ok=True)

    galaxies = _load_study_sample()
    if args.only:
        wanted = set(args.only.split(","))
        galaxies = [(s, k) for (s, k) in galaxies if k in wanted]
        missing = wanted - {k for _, k in galaxies}
        if missing:
            raise SystemExit(f"--only contained unknown lvdb_keys: {missing}")

    for study_name, _ in galaxies:
        wp = WEIGHT_DIR / f"{study_name}.npz"
        if not wp.exists():
            raise SystemExit(f"missing weight file for {study_name!r}: {wp}")

    print(f"loading halo catalog: {SATGEN_H5}")
    log10_rs_full, log10_rhos_full, in_range, _G = _load_halos()
    print(f"halos in range: {int(in_range.sum())}/{in_range.size}")

    summary = []
    for study_name, lvdb_key in galaxies:
        out_path = out_dir / f"{lvdb_key}.npz"
        npz_payload, meta = _build_for_dwarf(
            study_name, lvdb_key, args.shmr,
            log10_rs_full, log10_rhos_full, in_range,
        )
        np.savez(out_path, **npz_payload)
        summary.append(meta)
        print(
            f"  {lvdb_key:<22s}  n_nonzero={meta['n_halos_nonzero']:>7d}  "
            f"zero_frac={meta['zero_weight_fraction']:.3f}  "
            f"ESS={meta['ess']:>9.0f}  "
            f"n_bins={meta['n_bins']}  -> {out_path.name}"
        )

    summary_path = out_dir / "build_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
