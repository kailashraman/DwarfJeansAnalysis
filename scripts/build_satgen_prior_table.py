"""Derive the SatGen-based (r_s, ρ_s) prior table.

Reads the ΛCDM-with-tidal-stripping subhalo catalog from SatGen_Dwarf,
converts each halo's (v_max, r_max) to NFW (r_s, ρ_s), bins in
log10 r_s, and saves the per-bin Gaussian moments of log10 ρ_s plus
the marginal CDF of log10 r_s.

Outputs:
  data/satgen_prior/m12res8_diemer_scatter.npz
  data/satgen_prior/diagnostics_scatter.png
  data/satgen_prior/diagnostics_perbin_hist.png
  data/satgen_prior/diagnostics_qq.png
  data/satgen_prior/diagnostics_skew_kurt_N.png

The npz is the only artifact the inference path consumes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SATGEN_PATH = Path(
    "/global/scratch/projects/pc_heptheory/kraman/SatGen_Dwarf/data/additional/"
    "m12res8_10k_Diemer+scatter_sim.h5"
)
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "satgen_prior"

# Match the registry bounds.
LOG10_RS_BOUNDS = (-2.0, 1.0)
N_BINS = 30
N_CDF = 1024
MIN_BIN_COUNT = 200

# Derived from x_max ≈ 2.16258 solving the NFW v_c maximum.
RS_OVER_RMAX = 0.46241029979236
RHOS_COEFF = 1.7212585601570  # ρ_s = RHOS_COEFF · (v_max/r_max)² / G


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(SATGEN_PATH, "r") as fh:
        v_max_kpcGyr = fh["v_max"][()]
        r_max_kpc = fh["r_max"][()]

    v_max = (v_max_kpcGyr * u.kpc / u.Gyr).to(u.km / u.s).value
    G_val = const.G.to(u.km**2 * u.kpc / (u.s**2 * u.Msun)).value

    r_s = RS_OVER_RMAX * r_max_kpc
    rho_s = RHOS_COEFF * (v_max / r_max_kpc) ** 2 / G_val

    finite = np.isfinite(r_s) & np.isfinite(rho_s) & (r_s > 0) & (rho_s > 0)
    r_s = r_s[finite]
    rho_s = rho_s[finite]
    n_dropped = int((~finite).sum())

    log10_rs = np.log10(r_s)
    log10_rhos = np.log10(rho_s)

    in_range = (log10_rs >= LOG10_RS_BOUNDS[0]) & (log10_rs <= LOG10_RS_BOUNDS[1])
    log10_rs = log10_rs[in_range]
    log10_rhos = log10_rhos[in_range]
    n_outside = int((~in_range).sum())

    bin_edges = np.linspace(LOG10_RS_BOUNDS[0], LOG10_RS_BOUNDS[1], N_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mu_log10_rhos = np.full(N_BINS, np.nan)
    sigma_log10_rhos = np.full(N_BINS, np.nan)
    skew_per_bin = np.full(N_BINS, np.nan)
    kurt_per_bin = np.full(N_BINS, np.nan)
    n_per_bin = np.zeros(N_BINS, dtype=int)

    bin_idx = np.clip(
        np.digitize(log10_rs, bin_edges) - 1, 0, N_BINS - 1
    )
    for k in range(N_BINS):
        sel = bin_idx == k
        n_per_bin[k] = int(sel.sum())
        if n_per_bin[k] >= 2:
            y = log10_rhos[sel]
            mu_log10_rhos[k] = float(np.mean(y))
            sigma_log10_rhos[k] = float(np.std(y, ddof=1))
            if n_per_bin[k] >= 8:
                skew_per_bin[k] = float(stats.skew(y))
                kurt_per_bin[k] = float(stats.kurtosis(y, fisher=True))

    # Fill any low-count bins by nearest-neighbour interpolation of μ, σ over
    # the bins that meet MIN_BIN_COUNT — keeps the runtime interpolant defined
    # everywhere in LOG10_RS_BOUNDS even at the edges.
    good = n_per_bin >= MIN_BIN_COUNT
    if good.sum() < 2:
        raise RuntimeError(
            f"Too few bins with >={MIN_BIN_COUNT} halos: {good.sum()}"
        )
    mu_log10_rhos = np.interp(bin_centers, bin_centers[good], mu_log10_rhos[good])
    sigma_log10_rhos = np.interp(bin_centers, bin_centers[good], sigma_log10_rhos[good])

    # Marginal CDF of log10 r_s on a fine grid.
    sorted_rs = np.sort(log10_rs)
    cdf_y = np.linspace(0.0, 1.0, len(sorted_rs))
    log10_rs_grid = np.linspace(LOG10_RS_BOUNDS[0], LOG10_RS_BOUNDS[1], N_CDF)
    cdf_log10_rs = np.interp(log10_rs_grid, sorted_rs, cdf_y, left=0.0, right=1.0)
    # Enforce monotonicity and endpoints (np.interp already gives this, but
    # guard against ties producing tiny non-monotone steps).
    cdf_log10_rs = np.maximum.accumulate(cdf_log10_rs)
    cdf_log10_rs[0] = 0.0
    cdf_log10_rs[-1] = 1.0

    # Pooled residuals for the QQ plot.
    mu_at_x = np.interp(log10_rs, bin_centers, mu_log10_rhos)
    sigma_at_x = np.interp(log10_rs, bin_centers, sigma_log10_rhos)
    residuals = (log10_rhos - mu_at_x) / sigma_at_x
    pooled_skew = float(stats.skew(residuals))
    pooled_kurt = float(stats.kurtosis(residuals, fisher=True))

    metadata = {
        "source_path": str(SATGEN_PATH),
        "source_sha256": _sha256(SATGEN_PATH),
        "x_max": 1.0 / RS_OVER_RMAX,
        "rs_over_rmax": RS_OVER_RMAX,
        "rhos_coeff": RHOS_COEFF,
        "G_value": G_val,
        "G_units": "km^2 kpc / (s^2 Msun)",
        "n_total_in_h5": int(v_max_kpcGyr.size),
        "n_dropped_nonfinite": n_dropped,
        "n_outside_log10_rs_bounds": n_outside,
        "n_used": int(log10_rs.size),
        "log10_rs_bounds": list(LOG10_RS_BOUNDS),
        "n_bins": N_BINS,
        "n_cdf": N_CDF,
        "min_bin_count": MIN_BIN_COUNT,
        "pooled_skew": pooled_skew,
        "pooled_excess_kurtosis": pooled_kurt,
    }

    np.savez(
        OUT_DIR / "m12res8_diemer_scatter.npz",
        log10_rs_grid=log10_rs_grid,
        cdf_log10_rs=cdf_log10_rs,
        bin_centers_log10_rs=bin_centers,
        mu_log10_rhos=mu_log10_rhos,
        sigma_log10_rhos=sigma_log10_rhos,
        n_per_bin=n_per_bin,
        skew_per_bin=skew_per_bin,
        kurt_per_bin=kurt_per_bin,
        metadata=json.dumps(metadata),
    )

    # ----- diagnostics -----
    rng = np.random.default_rng(0)
    show = rng.choice(log10_rs.size, size=min(50_000, log10_rs.size), replace=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hexbin(log10_rs[show], log10_rhos[show], gridsize=80, mincnt=1, cmap="Greys")
    ax.plot(bin_centers, mu_log10_rhos, "r-", lw=1.5, label=r"$\mu(\log r_s)$")
    ax.fill_between(
        bin_centers,
        mu_log10_rhos - sigma_log10_rhos,
        mu_log10_rhos + sigma_log10_rhos,
        color="red", alpha=0.2, label=r"$\mu \pm \sigma$",
    )
    ax.set_xlabel(r"$\log_{10}(r_s/\mathrm{kpc})$")
    ax.set_ylabel(r"$\log_{10}(\rho_s/[M_\odot/\mathrm{kpc}^3])$")
    ax.set_title("SatGen NFW halos: per-bin Gaussian fit")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "diagnostics_scatter.png", dpi=120)
    plt.close(fig)

    # Per-bin histograms for a handful of bins.
    show_bins = np.linspace(0, N_BINS - 1, 6).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for ax_, k in zip(axes.flat, show_bins):
        sel = bin_idx == k
        if sel.sum() < 8:
            ax_.set_visible(False)
            continue
        y = log10_rhos[sel]
        ax_.hist(y, bins=40, density=True, color="lightgray", edgecolor="k")
        xs = np.linspace(y.min(), y.max(), 200)
        ax_.plot(
            xs,
            stats.norm.pdf(xs, mu_log10_rhos[k], sigma_log10_rhos[k]),
            "r-", lw=1.5,
        )
        ax_.set_title(
            f"bin {k}: log r_s∈[{bin_edges[k]:.2f},{bin_edges[k+1]:.2f}]  N={n_per_bin[k]}",
            fontsize=9,
        )
        ax_.set_xlabel(r"$\log_{10}\rho_s$")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "diagnostics_perbin_hist.png", dpi=120)
    plt.close(fig)

    # QQ plot of pooled standardized residuals.
    fig, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(residuals[rng.choice(residuals.size, 20_000, replace=False)],
                   dist="norm", plot=ax)
    ax.set_title(
        f"QQ pooled residuals  skew={pooled_skew:+.3f}  exkurt={pooled_kurt:+.3f}"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "diagnostics_qq.png", dpi=120)
    plt.close(fig)

    # Skew/kurtosis vs r_s + N.
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    axes[0].plot(bin_centers, skew_per_bin, "o-")
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set_ylabel("skew")
    axes[1].plot(bin_centers, kurt_per_bin, "o-")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_ylabel("excess kurtosis")
    axes[2].plot(bin_centers, n_per_bin, "o-")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("N halos / bin")
    axes[2].axhline(MIN_BIN_COUNT, color="r", lw=0.5, ls="--", label=f"MIN={MIN_BIN_COUNT}")
    axes[2].legend(fontsize=8)
    axes[2].set_xlabel(r"$\log_{10}(r_s/\mathrm{kpc})$")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "diagnostics_skew_kurt_N.png", dpi=120)
    plt.close(fig)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
