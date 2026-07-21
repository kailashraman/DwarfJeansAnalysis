#!/usr/bin/env python
"""Diagnose which dwarfs show a linear line-of-sight velocity gradient.

For each catalog we run `prepare_jeans_input` (same selection + multi-epoch
combination the Jeans fit uses) and fit a maximum-likelihood *linear* velocity
gradient across the projected face of the dwarf:

    v_i ~ N( v_sys + g_xi * xi_i + g_eta * eta_i ,  sigma_err_i^2 + sigma_int^2 )

with (xi, eta) the flat-sky East/North offsets in kpc from the registry center.
The mean parameters (v_sys, g_xi, g_eta) are profiled analytically by weighted
least squares; the one remaining scalar sigma_int is optimized by 1-D search.
Significance is the likelihood-ratio statistic vs. the g=0 null model
(2 d.o.f.), reported as an equivalent Gaussian sigma.

We fit twice per dwarf:
  * ``V_observed`` -- the raw velocities: the *apparent* gradient, which mixes
    intrinsic rotation/tidal signal with the perspective (PM-induced) gradient.
  * ``V``          -- the perspective-corrected velocities (see
    ``preprocess._apply_perspective``): the *intrinsic* residual gradient.

Dwarfs are ranked by the intrinsic (perspective-corrected) significance.

This is an exploratory diagnostic, not part of the Jeans likelihood. It does
not modify any pipeline output.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm

from dwarfjeans.jeans.preprocess import prepare_jeans_input
from dwarfjeans.postprocess import read_registry_row

REPO = Path(__file__).resolve().parents[1]
CATALOG_DIR = REPO / "data" / "star_catalogs"


def sky_offsets_kpc(ra, dec, ra0, dec0, dist_kpc):
    """Flat-sky East/North offsets in kpc, matching staging.projected_radius_kpc.

    East is increasing RA; North is increasing Dec.
    """
    dec0_rad = np.deg2rad(dec0)
    dra = np.deg2rad((ra - ra0) * np.cos(dec0_rad))  # eastward angle (rad)
    ddec = np.deg2rad(dec - dec0)                     # northward angle (rad)
    return dra * dist_kpc, ddec * dist_kpc


def _profile_negll(log_s, A, v, verr):
    """Profile negative log-likelihood at fixed intrinsic dispersion.

    Returns (negll, beta, cov_beta) where beta solves the weighted normal
    equations and cov_beta = (A^T W A)^{-1} is the Gaussian parameter
    covariance at the fitted mean.
    """
    s2 = np.exp(2.0 * log_s)
    w = 1.0 / (verr**2 + s2)
    AtW = A.T * w                      # (p, N)
    AtWA = AtW @ A                      # (p, p)
    cov = np.linalg.inv(AtWA)
    beta = cov @ (AtW @ v)
    resid = v - A @ beta
    negll = 0.5 * np.sum(w * resid**2) - 0.5 * np.sum(np.log(w)) \
        + 0.5 * v.size * np.log(2.0 * np.pi)
    return negll, beta, cov


def _fit_design(A, v, verr):
    """Minimize the profile negll over log(sigma_int) for design matrix A.

    ``railed`` is True when the optimizer parks sigma_int against either
    bracket bound. At the lower bound the Wald covariance ``cov`` is
    evaluated with near-zero intrinsic scatter and is overconfident, so the
    conditional |g| error derived from it must not be trusted (use the
    profile LRT significance instead).
    """
    # Bracket sigma_int between ~0 and a few x the raw velocity scatter.
    v_spread = np.std(v) + 1e-6
    lo, hi = np.log(1e-3), np.log(10.0 * v_spread)
    res = minimize_scalar(
        lambda ls: _profile_negll(ls, A, v, verr)[0],
        bounds=(lo, hi),
        method="bounded",
    )
    log_s = res.x
    negll, beta, cov = _profile_negll(log_s, A, v, verr)
    railed = bool(log_s < lo + 1e-3 or log_s > hi - 1e-3)
    return {
        "negll": negll,
        "beta": beta,
        "cov": cov,
        "sigma_int": float(np.exp(log_s)),
        "railed": railed,
    }


def fit_gradient(xi, eta, v, verr):
    """Fit the linear-gradient and null models; return a summary dict."""
    n = v.size
    ones = np.ones(n)
    A_full = np.column_stack([ones, xi, eta])
    A_null = ones[:, None]

    full = _fit_design(A_full, v, verr)
    null = _fit_design(A_null, v, verr)

    _, g_xi, g_eta = full["beta"]
    var_gxi = full["cov"][1, 1]
    var_geta = full["cov"][2, 2]
    cov_g = full["cov"][1, 2]

    g_mag = float(np.hypot(g_xi, g_eta))
    # Error propagation for |g| = sqrt(g_xi^2 + g_eta^2).
    if g_mag > 0:
        dxi, deta = g_xi / g_mag, g_eta / g_mag
        var_gmag = dxi**2 * var_gxi + deta**2 * var_geta + 2 * dxi * deta * cov_g
        sig_gmag = float(np.sqrt(max(var_gmag, 0.0)))
    else:
        sig_gmag = float(np.sqrt(max(var_gxi, var_geta, 0.0)))

    # Kinematic PA of the gradient (direction of increasing v), East of North.
    pa = float(np.rad2deg(np.arctan2(g_xi, g_eta)) % 360.0)

    # Interpretable across-face amplitude: center-to-edge half-amplitude of the
    # fitted trend at the 95th-percentile projected radius (robust to a single
    # outlier star). km/s/kpc alone is misleading for compact ultra-faints.
    r95 = float(np.percentile(np.hypot(xi, eta), 95))
    dv_halfamp = g_mag * r95        # km/s, center -> R95
    dv_edge2edge = 2.0 * dv_halfamp  # km/s across the field

    lrt = 2.0 * (null["negll"] - full["negll"])
    lrt = max(lrt, 0.0)
    pval = float(chi2.sf(lrt, df=2))
    # Two-sided Gaussian-equivalent sigma from the chi2(2) p-value.
    sigma = float(norm.isf(pval / 2.0)) if pval > 0 else np.inf

    # sig_gmag comes from full["cov"], so strictly only full["railed"] bears on
    # its reliability; we also fold in null["railed"] (conservative -- may
    # over-flag in the rare case where only the null nuisance rails) and quote
    # the LRT sigma whenever the flag fires.
    sigma_int_railed = bool(full["railed"] or null["railed"])

    return {
        "n": n,
        "g_mag": g_mag,          # km/s/kpc
        "r95_kpc": r95,
        "dv_halfamp": dv_halfamp,      # km/s, center -> R95
        "dv_edge2edge": dv_edge2edge,  # km/s across the field
        "sigma_int": full["sigma_int"],
        "sigma_int_railed": sigma_int_railed,
        "sig_gmag": sig_gmag,    # km/s/kpc
        "g_over_err": g_mag / sig_gmag if sig_gmag > 0 else np.inf,
        "pa_deg": pa,            # East of North
        "lrt": float(lrt),
        "pval": pval,
        "sigma": sigma,
    }


def run_one(key):
    cat = np.load(CATALOG_DIR / f"{key}.npz", allow_pickle=True)
    row = read_registry_row(key)
    filtered, audit = prepare_jeans_input(cat, row)

    ra = np.asarray(filtered["RA_star"], float)
    dec = np.asarray(filtered["Dec_star"], float)
    verr = np.asarray(filtered["sigma_eps"], float)
    v_corr = np.asarray(filtered["V"], float)
    v_obs = np.asarray(filtered.get("V_observed", filtered["V"]), float)

    good = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(verr) & (verr > 0) \
        & np.isfinite(v_corr) & np.isfinite(v_obs)
    ra, dec, verr = ra[good], dec[good], verr[good]
    v_corr, v_obs = v_corr[good], v_obs[good]

    persp = audit.get("perspective", {})
    result = {
        "key": key,
        "n": int(good.sum()),
        "persp_applied": bool(persp.get("applied", False)),
        "persp_rms_kms": float(persp.get("rms_kms", 0.0)),
    }
    if good.sum() < 10:
        result["skip"] = "fewer than 10 usable stars"
        return result

    xi, eta = sky_offsets_kpc(ra, dec, row["ra_deg"], row["dec_deg"],
                              row["distance_kpc"])
    result["intrinsic"] = fit_gradient(xi, eta, v_corr, verr)
    result["apparent"] = fit_gradient(xi, eta, v_obs, verr)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", help="lvdb keys (default: all catalogs)")
    ap.add_argument("--csv", type=Path, default=None, help="write ranked table")
    args = ap.parse_args()

    keys = args.keys or sorted(p.stem for p in CATALOG_DIR.glob("*.npz"))
    rows = []
    for key in keys:
        try:
            rows.append(run_one(key))
        except Exception as exc:  # noqa: BLE001 - diagnostic, keep going
            rows.append({"key": key, "error": repr(exc)})

    fitted = [r for r in rows if "intrinsic" in r]
    fitted.sort(key=lambda r: r["intrinsic"]["sigma"], reverse=True)

    hdr = (f"{'dwarf':<20} {'N':>5} {'|g|_intr':>9} {'+/-':>7} "
           f"{'dv_face':>7} {'sig':>5} {'PA':>5}  {'|g|_app':>9} {'app_sig':>7}  persp")
    print(hdr)
    print("-" * len(hdr))
    print("  |g| in km/s/kpc; dv_face = edge-to-edge km/s across the tracers; "
          "'*' after sig = sigma_int railed, |g| error unreliable -> trust sig.")
    for r in fitted:
        it, ap_ = r["intrinsic"], r["apparent"]
        flag = f"rms={r['persp_rms_kms']:.2f}" if r["persp_applied"] else "NOT applied"
        rail = "*" if it["sigma_int_railed"] else " "
        print(f"{r['key']:<20} {r['n']:>5} {it['g_mag']:>9.2f} "
              f"{it['sig_gmag']:>7.2f} {it['dv_edge2edge']:>7.2f} "
              f"{it['sigma']:>4.1f}{rail} {it['pa_deg']:>5.0f}  "
              f"{ap_['g_mag']:>9.2f} {ap_['sigma']:>7.1f}  {flag}")

    for r in rows:
        if "error" in r:
            print(f"  [error] {r['key']}: {r['error']}")
        elif "skip" in r:
            print(f"  [skip]  {r['key']}: {r['skip']} (N={r['n']})")

    if args.csv:
        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            # Column names: `significance_*` are LRT sigma (detection strength);
            # `sigma_int_kms` is the fitted intrinsic velocity dispersion (a
            # nuisance, in km/s) -- kept distinct to avoid a sigma/sigma_int mixup.
            w.writerow(["dwarf", "n", "persp_applied", "persp_rms_kms",
                        "g_mag_intrinsic", "sig_g_intrinsic",
                        "dv_edge2edge_intrinsic", "sigma_int_kms",
                        "sigma_int_railed", "significance_intrinsic",
                        "pa_intrinsic", "g_mag_apparent",
                        "significance_apparent"])
            for r in fitted:
                it, ap_ = r["intrinsic"], r["apparent"]
                w.writerow([r["key"], r["n"], r["persp_applied"],
                            f"{r['persp_rms_kms']:.4f}",
                            f"{it['g_mag']:.4f}", f"{it['sig_gmag']:.4f}",
                            f"{it['dv_edge2edge']:.4f}", f"{it['sigma_int']:.4f}",
                            it["sigma_int_railed"], f"{it['sigma']:.3f}",
                            f"{it['pa_deg']:.1f}", f"{ap_['g_mag']:.4f}",
                            f"{ap_['sigma']:.3f}"])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
