"""
Quick Segue 1 Jeans run using docs/original-plan scripts.

Reads global Segue 1 properties from LVDB v1.0.5 (cached locally),
applies a membership cut on the per-star catalog, runs Stage-2 dynesty
inference via jeans_inference.run_inference, then derives σ_los, M_half,
J, D posterior chains and writes plots + summary.

Per-star source options (SOURCE toggle):
  'simon': Simon+2011 VizieR Table 1, Bpr > 0.8
  'pace':  Pace pre-combined Bayes 0.8 file
  'geha':  Geha+2026 Table 5A (Pmem_novar==1) + sphericalized 3D Plummer
           half-mass cut R < 2·rhalf_major·√(1−ε)·(4/3) per the production
           SelectionPolicy. Uses LVDB nuisance priors instead of PS18 overrides.
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_DIR = REPO / "results" / "tests" / "segue1"

from dwarfjeans.jeans import solver as jeans
from dwarfjeans.jeans import inference as jeans_inference
from dwarfjeans.jeans.preprocess import prepare_jeans_input
from dwarfjeans.jeans.selection import SelectionPolicy
from dwarfjeans.jd import factors as jdf
from dwarfjeans.jeans.constant_sigma import constant_sigma_inference as _pkg_constant_sigma_inference

LVDB_URL = ("https://github.com/apace7/local_volume_database/"
            "releases/download/v1.0.5/comb_all.csv")
LVDB_CACHE     = REPO / "data" / "lvdb_v1.0.5" / "comb_all.csv"
SEGUE1_KIN_CSV = REPO / "data" / "segue1" / "segue1_kinematics_simon2011.csv"
PACE_DAT       = REPO / "data" / "segue1" / "Pace_Segue1_Bayes_0d8_binary.dat"
P_CUT = 0.8
P_CUT_GEHA = 0.5  # Geha+2026 §3.1 sample selection
ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)
PLUMMER_3D_OVER_2D = 1.30477  # r_½(3D) / r_½(2D) for Plummer

# Per-star data source toggle.
#   'simon': segue1_kinematics_simon2011.csv (VizieR Simon+2011 Table 1) with Bpr > 0.8.
#   'pace':  Pace_Segue1_Bayes_0d8_binary.dat (Pace 0.8-membership combined-velocity file).
#            simon/pace select the same 62 stars (compare_pace_vs_bpr08.py); both use PS18 priors.
#   'geha':  data/star_catalogs/segue_1.npz (staged from Geha+2026 Table 5A, Pmem_novar);
#            production SelectionPolicy(p_min=0.5, R_over_rhalf_max=2.0, drop_variable=True)
#            with sphericalized 3D Plummer cut → 52 stars.
#            Uses LVDB v1.0.5 nuisance priors (d, rhalf, ε) instead of PS18 overrides.
SOURCE = "geha"  # 'simon' | 'pace' | 'geha'
USE_P_WEIGHTS = False     # if False, replace post-cut p_i with 1.0 in the likelihood
PRIOR_NAME = "loguniform"    # one of {"uniform", "loguniform", "jeffreys"} from dwarfjeans.jeans.priors
DYNESTY_NLIVE = 500   # nominal 500; raise for posterior stability
DYNESTY_DLOGZ = 0.1   # nominal 0.1; tighten alongside nlive
# When set to (mean, sigma), bypass the rhalf×√(1−ε) chain in the Jeans
# likelihood and use a direct Normal prior on the angular Plummer scale
# r_p_arcmin instead. ε is still sampled but unused for r_p geometry.
# FIX_R_P_ARCMIN: tuple[float, float] | None = (4.49, 0.85)
FIX_R_P_ARCMIN: tuple[float, float] | None = None

# Pace+Strigari 2018 fixed observational parameters for Segue 1
PS18_D_KPC        = 23.0   # distance [kpc]
PS18_D_KPC_ERR    = 2.0    # 1-sigma uncertainty [kpc]
PS18_RHALF_PC     = 21.0   # azimuthally averaged half-light radius [pc]
PS18_RHALF_PC_ERR = 5.0    # 1-sigma uncertainty [pc]


def fetch_lvdb() -> Path:
    LVDB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not LVDB_CACHE.exists():
        urllib.request.urlretrieve(LVDB_URL, LVDB_CACHE)
    return LVDB_CACHE


def load_segue1_lvdb() -> dict:
    df = pd.read_csv(fetch_lvdb())
    row = df[df["key"] == "segue_1"]
    if len(row) != 1:
        raise RuntimeError(f"Segue 1 lookup failed: {len(row)} rows")
    r = row.iloc[0]

    rhalf_arcmin = float(r["rhalf"])
    eps = float(r["ellipticity"]) if not pd.isna(r["ellipticity"]) else 0.0
    mu = float(r["distance_modulus"])
    d_kpc = 10.0 ** (1.0 + mu / 5.0) / 1000.0  # pc -> kpc
    R_half_2d = d_kpc * rhalf_arcmin * ARCMIN_TO_RAD * np.sqrt(1.0 - eps)
    # Segue 1: assumed Plummer (Martin+2008). LVDB v1.0.5 has no explicit
    # spatial_model column; logged so the assumption is visible.
    r_p = R_half_2d
    r_half_3d = PLUMMER_3D_OVER_2D * r_p
    V_center = float(r["vlos_systemic"]) if not pd.isna(r["vlos_systemic"]) else 208.5

    return {
        "rhalf_arcmin": rhalf_arcmin,
        "ellipticity": eps,
        "distance_modulus": mu,
        "d_kpc": d_kpc,
        "R_half_2d_kpc": R_half_2d,
        "r_p_kpc": r_p,
        "r_half_3d_kpc": r_half_3d,
        "V_center_kms": V_center,
        "ra_deg":  float(r["ra"]),
        "dec_deg": float(r["dec"]),
        "vlos_sigma_lvdb_kms": float(r["vlos_sigma"]) if not pd.isna(r["vlos_sigma"]) else np.nan,
    }


def load_stars_simon() -> pd.DataFrame:
    """
    Read the Simon+2011 catalog and inverse-variance-combine multi-epoch
    measurements per unique star (97 of 393 stars have 2-3 epochs across
    15 distinct MJDs; only one epoch per star carries the Bpr membership
    tag, the others are NaN). Combining gives one row per star with
        V_comb = sum(V_t/eV_t^2) / sum(1/eV_t^2)
        eV_comb = 1 / sqrt(sum(1/eV_t^2))
    matching the operational convention in Pace's Bayes_0d8_binary.dat.
    """
    df = pd.read_csv(SEGUE1_KIN_CSV)
    df = df.dropna(subset=["Vel", "e_Vel", "_RA", "_DE", "SDSS"])

    rows = []
    for sid, g in df.groupby("SDSS"):
        # Catalog should be self-consistent on position per SDSS id; defend.
        if (g["_RA"].max() - g["_RA"].min()) * 3600.0 > 1.0 \
           or (g["_DE"].max() - g["_DE"].min()) * 3600.0 > 1.0:
            raise ValueError(
                f"SDSS id {sid} has rows whose (_RA, _DE) disagree by > 1\". "
                "Refusing to combine."
            )
        bpr_vals = g["Bpr"].dropna().unique()
        if len(bpr_vals) > 1:
            raise ValueError(
                f"SDSS id {sid} has multiple distinct non-NaN Bpr values: "
                f"{bpr_vals.tolist()}."
            )
        bpr = float(bpr_vals[0]) if len(bpr_vals) == 1 else np.nan

        Vs = g["Vel"].values
        eVs = g["e_Vel"].values
        w = 1.0 / eVs ** 2
        V_comb = float(np.sum(Vs * w) / np.sum(w))
        eV_comb = float(np.sqrt(1.0 / np.sum(w)))
        rad = float(g["Rad"].iloc[0])
        ra0 = float(g["_RA"].iloc[0])
        dec0 = float(g["_DE"].iloc[0])

        rows.append({
            "SDSS": sid,
            "Vel": V_comb, "e_Vel": eV_comb, "Bpr": bpr,
            "Rad": rad, "Rad_arcmin": rad,
            "_RA": ra0, "_DE": dec0,
            "n_epochs": int(len(g)),
        })

    keep = pd.DataFrame(rows).dropna(subset=["Bpr"])
    keep = keep[keep["Bpr"] > P_CUT].reset_index(drop=True)
    return keep


def load_stars_pace(ra_center_deg: float, dec_center_deg: float) -> pd.DataFrame:
    """
    Load Pace's `Bayes_0d8_binary.dat` (62 stars, all p ≥ 0.8).
    Columns: RA[deg], Dec[deg], V[km/s], e_V[km/s], Bayes-membership p.
    Computes Rad_arcmin from sky position relative to the LVDB galaxy center
    (small-angle; Segue 1 spans ≲ 6 arcmin so the flat-sky approximation is
    well within per-star astrometric error).
    """
    P = np.loadtxt(PACE_DAT, skiprows=1)
    ra, dec, V, eV, p = P.T
    cos_d = np.cos(np.radians(dec_center_deg))
    dRA = (ra - ra_center_deg) * cos_d
    dDec = dec - dec_center_deg
    rad_deg = np.sqrt(dRA ** 2 + dDec ** 2)
    rad_arcmin = rad_deg * 60.0
    keep = pd.DataFrame({
        "Vel": V, "e_Vel": eV, "Bpr": p, "Rad_arcmin": rad_arcmin,
        "Rad": rad_arcmin,  # kept for schema parity with the Simon path
        "_RA": ra, "_DE": dec,
        "n_epochs": np.ones(len(V), dtype=int),  # Pace file is pre-combined
    })
    keep = keep[keep["Bpr"] > P_CUT].reset_index(drop=True)
    return keep


def load_stars_geha(d_kpc: float, registry_row: dict) -> tuple[pd.DataFrame, dict]:
    """
    Route the Segue 1 Geha-staged catalog through the analysis-time
    preprocessing pipeline (jeans.preprocess.prepare_jeans_input):
      - per-star granularity → no combiner
      - SelectionPolicy(p_min=P_CUT_GEHA, R_over_rhalf_max=2.0,
        drop_variable=True) matches the production §3.1 recipe:
        Pmem_novar==1 and R < 2·rhalf_major·√(1−ε)·(4/3) on the
        sphericalized 3D Plummer half-mass radius.

    Selection requires the full registry row (uses ``rhalf_major_pc``
    AND ``ellipticity``); pass via _read_registry_row from
    run_production. Returns the per-star DataFrame in the legacy schema
    downstream callers expect, plus the audit dict from
    prepare_jeans_input so the selection report can be logged and dumped.
    """
    catalog_path = REPO / "data" / "star_catalogs" / "segue_1.npz"
    cat = np.load(catalog_path, allow_pickle=True)
    arrays, audit = prepare_jeans_input(
        cat, registry_row,
        selection_policy=SelectionPolicy(
            p_min=P_CUT_GEHA,
            R_over_rhalf_max=2.0,
            drop_variable=True,
        ),
    )
    R_kpc = np.asarray(arrays["R"], dtype=float)
    Rad_arcmin = R_kpc / d_kpc / ARCMIN_TO_RAD
    df = pd.DataFrame({
        "Vel":        np.asarray(arrays["V"], dtype=float),
        "e_Vel":      np.asarray(arrays["sigma_eps"], dtype=float),
        "Bpr":        np.asarray(arrays["p"], dtype=float),
        "Rad_arcmin": Rad_arcmin,
        "Rad":        Rad_arcmin,
        "_RA":        np.asarray(arrays.get("RA_star", np.full(R_kpc.size, np.nan)),
                                  dtype=float),
        "_DE":        np.asarray(arrays.get("Dec_star", np.full(R_kpc.size, np.nan)),
                                  dtype=float),
        "n_epochs":   np.ones(len(R_kpc), dtype=int),
        "R_kpc":      R_kpc,
    })
    return df, audit


def lvdb_nuisance_priors(row: pd.Series) -> dict:
    """
    Extract symmetrised Gaussian nuisance priors from an LVDB v1.0.5 galaxy row.
    Propagates distance-modulus uncertainty to d_kpc: σ_d = (ln10/5) × d × σ_μ.
    Falls back to 5% of the central value if _em/_ep are NaN.
    """
    def sym(val, em_col, ep_col):
        em = row.get(em_col)
        ep = row.get(ep_col)
        if pd.isna(em) or pd.isna(ep):
            print(f"WARNING: {em_col}/{ep_col} NaN; using 5% of central value.")
            return 0.05 * abs(val)
        return (float(em) + float(ep)) / 2.0

    mu    = float(row["distance_modulus"])
    d_kpc = 10.0 ** (1.0 + mu / 5.0) / 1000.0
    mu_sig = sym(mu, "distance_modulus_em", "distance_modulus_ep")
    d_sig  = (np.log(10.0) / 5.0) * d_kpc * mu_sig

    rh     = float(row["rhalf"])
    rh_sig = sym(rh, "rhalf_em", "rhalf_ep")

    eps     = float(row["ellipticity"]) if not pd.isna(row["ellipticity"]) else 0.0
    eps_sig = sym(eps if eps != 0.0 else 0.1, "ellipticity_em", "ellipticity_ep")

    return {"d_mean": d_kpc,   "d_sigma":     d_sig,
            "eps_mean": eps,   "eps_sigma":   eps_sig,
            "rhalf_mean": rh,  "rhalf_sigma": rh_sig}


def load_stars(d_kpc: float, source: str,
                ra_center_deg: float | None = None,
                dec_center_deg: float | None = None,
                registry_row: dict | None = None,
                ) -> tuple[pd.DataFrame, dict | None]:
    audit: dict | None = None
    if source == "simon":
        keep = load_stars_simon()
    elif source == "pace":
        if ra_center_deg is None or dec_center_deg is None:
            raise ValueError("source='pace' requires ra/dec center")
        keep = load_stars_pace(ra_center_deg, dec_center_deg)
    elif source == "geha":
        if registry_row is None:
            raise ValueError("source='geha' requires registry_row")
        keep, audit = load_stars_geha(d_kpc, registry_row)
    else:
        raise ValueError(f"unknown source {source!r}")
    if "R_kpc" not in keep.columns:
        R_kpc = d_kpc * keep["Rad_arcmin"].values * ARCMIN_TO_RAD
        keep["R_kpc"] = R_kpc
    # The grid integrator in jeans.sigma_los uses u_min = 1e-4 * R, which
    # blows up at exactly R=0. Floor at a small positive value (1e-5 kpc
    # ≈ 1.5e-3 arcmin at d=23 kpc — far below per-star astrometric error).
    keep["R_kpc"] = np.clip(keep["R_kpc"].values, 1e-5, None)
    return keep, audit


constant_sigma_inference = _pkg_constant_sigma_inference  # re-export from package



def percentiles(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    q16, q50, q84 = np.percentile(arr, [16.0, 50.0, 84.0])
    return {"median": q50, "q16": q16, "q84": q84,
            "sigma_lo": q50 - q16, "sigma_hi": q84 - q50}


def main():
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir
    _suffix_parts = []
    if SOURCE != "simon":
        _suffix_parts.append(SOURCE)
    if not USE_P_WEIGHTS:
        _suffix_parts.append("nop")
    if SOURCE in ("simon", "pace") and FIX_R_P_ARCMIN is not None:
        _suffix_parts.append("fixrp")
    if PRIOR_NAME != "jeffreys":
        _suffix_parts.append("logunif" if PRIOR_NAME == "loguniform" else PRIOR_NAME)
    suffix = "_" + "_".join(_suffix_parts) if _suffix_parts else ""

    def out(stem: str, ext: str) -> Path:
        return out_dir / f"{stem}{suffix}.{ext}"
    log = []

    def logp(*a):
        msg = " ".join(str(x) for x in a)
        print(msg)
        log.append(msg)

    t0 = time.time()
    g = load_segue1_lvdb()
    logp("=== LVDB v1.0.5 Segue 1 ===")
    for k, v in g.items():
        logp(f"  {k}: {v}")

    if SOURCE in ("simon", "pace"):
        # Override LVDB values with Pace+Strigari 2018 observational parameters
        g["d_kpc"]         = PS18_D_KPC
        g["R_half_2d_kpc"] = PS18_RHALF_PC / 1000.0
        g["r_p_kpc"]       = PS18_RHALF_PC / 1000.0
        g["r_half_3d_kpc"] = PLUMMER_3D_OVER_2D * PS18_RHALF_PC / 1000.0
        logp("\n=== P&S 2018 observational overrides ===")
        logp(f"  d_kpc:         {g['d_kpc']:.1f} ± {PS18_D_KPC_ERR:.1f} kpc")
        logp(f"  r_p_kpc:       {g['r_p_kpc']*1000:.1f} ± {PS18_RHALF_PC_ERR:.1f} pc")
        logp(f"  r_half_3d_kpc: {g['r_half_3d_kpc']*1000:.2f} pc")
    else:
        logp("\n=== Using LVDB v1.0.5 observational values (no PS18 override) ===")
        logp(f"  d_kpc:        {g['d_kpc']:.2f} kpc")
        logp(f"  r_p_kpc:      {g['r_p_kpc']*1000:.2f} pc")
        logp(f"  r_half_3d:    {g['r_half_3d_kpc']*1000:.2f} pc")
        logp(f"  rhalf_arcmin: {g['rhalf_arcmin']:.3f}, ellipticity: {g['ellipticity']:.3f}")
        # Sanity check: d × rhalf_major × ARCMIN_TO_RAD should be ~0.024 kpc for Segue 1
        rhalf_major_kpc = g["d_kpc"] * g["rhalf_arcmin"] * ARCMIN_TO_RAD
        logp(f"  d × rhalf_major × ARCMIN_TO_RAD = {rhalf_major_kpc:.4f} kpc  (expect ~0.024)")
        if not (0.018 < rhalf_major_kpc < 0.035):
            raise RuntimeError(
                f"LVDB rhalf sanity check failed: d×rhalf_major = {rhalf_major_kpc:.4f} kpc, "
                "expected 0.018–0.035 kpc for Segue 1."
            )

    # Pull the full registry row — selection needs both rhalf_major_pc AND
    # ellipticity (post-2026-05-08). Same row staging.projected_radius_kpc
    # used to bake R_kpc into the .npz, so the cut and the radii can't drift.
    import sys as _sys
    _sys.path.insert(0, str(REPO / "scripts"))
    from run_production import _read_registry_row  # noqa: E402
    _registry_row = _read_registry_row("segue_1")
    _rhalf_major_pc = float(_registry_row["rhalf_major_pc"])
    stars, geha_audit = load_stars(g["d_kpc"], SOURCE,
                        ra_center_deg=g["ra_deg"], dec_center_deg=g["dec_deg"],
                        registry_row=_registry_row)
    src_name = (SEGUE1_KIN_CSV.name if SOURCE == "simon"
                else "data/star_catalogs/segue_1.npz" if SOURCE == "geha"
                else PACE_DAT.name)
    logp(f"\n=== Per-star data ({src_name}) ===")
    logp(f"  source: {SOURCE!r}")
    _pmem_cut = P_CUT_GEHA if SOURCE == "geha" else P_CUT
    _cut_desc = (f"Pmem>{_pmem_cut}, R<2×rhalf_major_pc={_rhalf_major_pc:.3f}pc "
                 f"(from data/registry/galaxies.ecsv), drop Var==1"
                 if SOURCE == "geha" else f"Bpr > {_pmem_cut}")
    logp(f"  {_cut_desc}: N = {len(stars)}")
    if geha_audit is not None:
        sel = geha_audit["selection"]
        logp(f"  selection report: n_input={sel['n_input']} -> "
             f"n_after_p={sel['n_after_p']} -> n_after_R={sel['n_after_R']} -> "
             f"n_after_var={sel['n_after_var']} -> n_final={sel['n_final']}")
    logp(f"  R range (kpc): [{stars['R_kpc'].min():.4f}, {stars['R_kpc'].max():.4f}]")
    logp(f"  V mean/std (km/s): {stars['Vel'].mean():.2f} / {stars['Vel'].std():.2f}")
    logp(f"  e_Vel median (km/s): {stars['e_Vel'].median():.2f}")
    logp(f"  p (Bpr) min/median: {stars['Bpr'].min():.3f} / {stars['Bpr'].median():.3f}")
    if "n_epochs" in stars.columns:
        ne = stars["n_epochs"].values
        logp(f"  multi-epoch stars: {(ne > 1).sum()}/{len(stars)}; "
             f"mean n_epochs = {ne.mean():.2f} (max {ne.max()})")

    p_raw = stars["Bpr"].values
    p_eff = p_raw if USE_P_WEIGHTS else np.ones_like(p_raw)
    if USE_P_WEIGHTS:
        logp(f"  USE_P_WEIGHTS=True: continuous p_i propagated into the likelihood")
    else:
        logp(f"  USE_P_WEIGHTS=False: post-cut p_i replaced by 1.0 in the likelihood")

    galaxy = {
        "R": stars["R_kpc"].values,                # used by 4D path; stale-d snapshot
        "Rad_arcmin": stars["Rad_arcmin"].values,  # used by 7D nuisance-marginalized path
        "V": stars["Vel"].values,
        "sigma_eps": stars["e_Vel"].values,
        "p": p_eff,
        "truth": {"r_p": g["r_p_kpc"]},
    }

    if SOURCE in ("simon", "pace"):
        # Nuisance priors: P&S 2018 distance + Martin+2008 ellipticity; rhalf set
        # so that at fiducial (d=23, ε=0.47) the implied r_p is 21 ± 5 pc per P&S 2018.
        nuisance_priors = {
            "d_mean":      23.0,    "d_sigma":      2.0,    # kpc
            "eps_mean":    0.47,    "eps_sigma":    0.11,   # dimensionless, truncated to [0,1)
            "rhalf_mean":  4.31,    "rhalf_sigma":  1.03,   # arcmin
        }
        if FIX_R_P_ARCMIN is not None:
            # The 7th nuisance is reinterpreted as r_p_arcmin directly; bypasses
            # √(1−ε) inside the likelihood. ε is still sampled (its own prior is
            # unchanged) but does not enter r_p geometry.
            nuisance_priors["rhalf_mean"]  = float(FIX_R_P_ARCMIN[0])
            nuisance_priors["rhalf_sigma"] = float(FIX_R_P_ARCMIN[1])
        fix_r_p = FIX_R_P_ARCMIN
    else:
        # Geha run: derive nuisance priors from LVDB v1.0.5 directly.
        lvdb_df = pd.read_csv(fetch_lvdb())
        lvdb_row = lvdb_df[lvdb_df["key"] == "segue_1"].iloc[0]
        nuisance_priors = lvdb_nuisance_priors(lvdb_row)
        fix_r_p = None  # sample rhalf_arcmin + eps separately; r_p derived in chain

    logp("\n=== Constant-σ inference ===")
    cs = constant_sigma_inference(
        galaxy["V"], galaxy["sigma_eps"], galaxy["p"],
        V_center=g["V_center_kms"],
    )
    logp(f"  V_sys:     {cs['V_sys']['median']:+.4g}  "
         f"[{cs['V_sys']['q16']:+.4g}, {cs['V_sys']['q84']:+.4g}]  (km/s)")
    logp(f"  sigma_int: {cs['sigma_int']['median']:.4g}  "
         f"[{cs['sigma_int']['q16']:.4g}, {cs['sigma_int']['q84']:.4g}]  (km/s)")

    logp(f"\n=== Running dynesty (7D, nuisance-marginalized; nlive={DYNESTY_NLIVE}, dlogz={DYNESTY_DLOGZ}) ===")
    _rp_label = "r_p_arcmin (fix_r_p_arcmin)" if fix_r_p is not None else "rhalf"
    logp(f"  nuisance priors: d ~ N({nuisance_priors['d_mean']}, {nuisance_priors['d_sigma']}) kpc, "
         f"ε ~ N({nuisance_priors['eps_mean']}, {nuisance_priors['eps_sigma']}) trunc [0,1), "
         f"{_rp_label} ~ N({nuisance_priors['rhalf_mean']}, {nuisance_priors['rhalf_sigma']}) arcmin")
    logp("  halo prior: conditional Jeffreys on (ln ρ_s, ln r_s) at fixed β "
         "(jeffreys_jeans_derivation.md), truncated to LOG10_RS/RHOS_BOUNDS")
    t_inf = time.time()
    result = jeans_inference.run_inference(
        galaxy,
        V_center=g["V_center_kms"],
        nlive=DYNESTY_NLIVE,
        dlogz=DYNESTY_DLOGZ,
        rseed=0,
        print_progress=False,
        marginalize_nuisances=True,
        nuisance_priors=nuisance_priors,
        prior_name=PRIOR_NAME,
        fix_r_p_arcmin=(fix_r_p is not None),
    )
    logp(f"  done in {time.time()-t_inf:.1f}s")
    logp(f"  logZ = {result['logz']:.3f} ± {result['logz_err']:.3f}")
    logp(f"  n_eq = {result['n_eq']}")

    samples_eq = result["samples_eq"]
    V_chain, lr_chain, lp_chain, btilde_chain, d_chain, eps_chain, rhalf_arcmin_chain = samples_eq.T
    r_s_chain = 10.0 ** lr_chain
    rho_s_chain = 10.0 ** lp_chain
    beta_chain = jeans_inference.beta_tilde_to_beta(btilde_chain)

    # Per-sample derived nuisance chains
    if fix_r_p is not None:
        # 7th param is r_p_arcmin directly; matches the likelihood's geometry.
        r_p_chain = d_chain * rhalf_arcmin_chain * ARCMIN_TO_RAD
    else:
        r_p_chain = d_chain * rhalf_arcmin_chain * ARCMIN_TO_RAD * np.sqrt(1.0 - eps_chain)
    R_half_2d_chain  = r_p_chain                                # Plummer assumption
    r_half_3d_chain  = PLUMMER_3D_OVER_2D * r_p_chain
    r_t_kpc = 1.0  # placeholder; matches stage3.md mock-test default

    # Enclosed mass chains (vectorized — nfw_M is closed-form, and accepts arrays)
    log10_M_2d = np.log10(jeans.nfw_M(R_half_2d_chain, r_s_chain, rho_s_chain))
    log10_M_3d = np.log10(jeans.nfw_M(r_half_3d_chain, r_s_chain, rho_s_chain))

    # Sigma_los at R_half_2d — loop over chain (sigma_los takes scalar r_s, rho_s)
    N_chain = samples_eq.shape[0]
    THIN_TO = 2000  # σ_los is cheap; J/D is expensive
    rng = np.random.default_rng(0)
    if N_chain > THIN_TO:
        idx_sigma = rng.choice(N_chain, size=THIN_TO, replace=False)
    else:
        idx_sigma = np.arange(N_chain)

    sigma_at_Rhalf = np.empty(idx_sigma.size)
    for j, i in enumerate(idx_sigma):
        try:
            s = jeans.sigma_los(np.array([R_half_2d_chain[i]]), beta_chain[i],
                                 r_s_chain[i], rho_s_chain[i], r_p_chain[i],
                                 method="grid")
            sigma_at_Rhalf[j] = float(s[0])
        except Exception:
            sigma_at_Rhalf[j] = np.nan

    # Sigma_los profile chain (median + 68% band over a radial grid). The grid is
    # set from the empirical R range under the *posterior-median* d (so the radial
    # axis has a fixed meaning across draws).
    d_med = float(np.median(d_chain))
    R_kpc_med = d_med * stars["Rad_arcmin"].values * ARCMIN_TO_RAD
    R_grid = np.geomspace(max(np.clip(R_kpc_med, 1e-5, None).min(), 1e-3),
                            R_kpc_med.max(), 30)
    THIN_PROFILE = 300
    if N_chain > THIN_PROFILE:
        idx_prof = rng.choice(N_chain, size=THIN_PROFILE, replace=False)
    else:
        idx_prof = np.arange(N_chain)
    sigma_profile = np.full((idx_prof.size, R_grid.size), np.nan)
    for j, i in enumerate(idx_prof):
        try:
            sigma_profile[j] = jeans.sigma_los(R_grid, beta_chain[i],
                                                 r_s_chain[i], rho_s_chain[i],
                                                 r_p_chain[i], method="grid")
        except Exception:
            pass
    sig_lo = np.nanpercentile(sigma_profile, 16, axis=0)
    sig_med = np.nanpercentile(sigma_profile, 50, axis=0)
    sig_hi = np.nanpercentile(sigma_profile, 84, axis=0)

    # J/D chains: thin + loop
    THIN_JD = 500
    if N_chain > THIN_JD:
        idx_jd = rng.choice(N_chain, size=THIN_JD, replace=False)
    else:
        idx_jd = np.arange(N_chain)

    # alpha_c is per-sample (depends on per-draw d, r_half_3d). Fixed-angle entries are scalar.
    alpha_c_chain = jdf.alpha_c_radians(r_half_3d_chain, d_chain)
    fixed_angles_J = {
        "0p1deg": 0.1 * jdf.DEG,
        "0p2deg": 0.2 * jdf.DEG,
        "0p5deg": 0.5 * jdf.DEG,
    }
    fixed_angles_D = dict(fixed_angles_J)  # same fixed angles for D
    log10_J = {tag: np.empty(idx_jd.size) for tag in (*fixed_angles_J, "alphac")}
    log10_D = {tag: np.empty(idx_jd.size) for tag in (*fixed_angles_D, "alphacover2")}
    logp(f"\n=== J/D integrals (thin {idx_jd.size}/{N_chain}) ===")
    logp(f"  alpha_c (median): {float(np.median(alpha_c_chain)):.5f} rad "
         f"({float(np.median(alpha_c_chain))/jdf.DEG:.4f} deg)")
    t_jd = time.time()
    for j, i in enumerate(idx_jd):
        rs_i, rhos_i = float(r_s_chain[i]), float(rho_s_chain[i])
        d_i = float(d_chain[i])
        ac_i = float(alpha_c_chain[i])
        for tag, th in fixed_angles_J.items():
            J, _ = jdf.J_D_factors(th, d_i, rs_i, rhos_i, r_t_kpc)
            log10_J[tag][j] = np.log10(J) + jdf.LOG10_J_FAC if J > 0 else np.nan
        J_ac, _ = jdf.J_D_factors(ac_i, d_i, rs_i, rhos_i, r_t_kpc)
        log10_J["alphac"][j] = np.log10(J_ac) + jdf.LOG10_J_FAC if J_ac > 0 else np.nan
        for tag, th in fixed_angles_D.items():
            _, D = jdf.J_D_factors(th, d_i, rs_i, rhos_i, r_t_kpc)
            log10_D[tag][j] = np.log10(D) + jdf.LOG10_D_FAC if D > 0 else np.nan
        _, D_aco2 = jdf.J_D_factors(0.5 * ac_i, d_i, rs_i, rhos_i, r_t_kpc)
        log10_D["alphacover2"][j] = np.log10(D_aco2) + jdf.LOG10_D_FAC if D_aco2 > 0 else np.nan
    logp(f"  done in {time.time()-t_jd:.1f}s")

    # ---- Save samples ----
    np.savez(
        out("posterior_samples", "npz"),
        samples_eq=samples_eq,
        V=V_chain, log10_rs=lr_chain, log10_rhos=lp_chain,
        beta_tilde=btilde_chain, beta=beta_chain,
        d_kpc_chain=d_chain, eps_chain=eps_chain, rhalf_arcmin_chain=rhalf_arcmin_chain,
        r_p_chain=r_p_chain, R_half_2d_chain=R_half_2d_chain,
        r_half_3d_chain=r_half_3d_chain, alpha_c_chain=alpha_c_chain,
        log10_M_half_2d=log10_M_2d, log10_M_half_3d=log10_M_3d,
        sigma_los_at_Rhalf2d=sigma_at_Rhalf,
        R_grid_sigma_profile=R_grid,
        sigma_profile_q16=sig_lo, sigma_profile_q50=sig_med, sigma_profile_q84=sig_hi,
        **{f"log10_J_{tag}": v for tag, v in log10_J.items()},
        **{f"log10_D_{tag}": v for tag, v in log10_D.items()},
        r_t_kpc=r_t_kpc,
        sigma_los_walker_V_grid=cs["V_grid"],
        sigma_los_walker_sigma_grid=cs["sigma_grid"],
        sigma_los_walker_log10_sigma_grid=cs["log10_sigma_grid"],
        sigma_los_walker_marg_V=cs["marg_V"],
        sigma_los_walker_marg_sigma=cs["marg_sigma"],
        sigma_los_walker_marg_log10_sigma=cs["marg_log10_sigma"],
    )

    # ---- Summary CSV ----
    summary_rows = []
    def addrow(name, arr, unit=""):
        s = percentiles(arr)
        summary_rows.append({"quantity": name, "unit": unit, **s})

    addrow("V_sys", V_chain, "km/s")
    addrow("log10_rs", lr_chain, "log10(kpc)")
    addrow("log10_rhos", lp_chain, "log10(Msun/kpc^3)")
    addrow("beta_tilde", btilde_chain, "")
    addrow("beta", beta_chain, "")
    addrow("d_kpc",         d_chain,             "kpc")
    addrow("eps",           eps_chain,           "")
    addrow("rhalf_arcmin",  rhalf_arcmin_chain,  "arcmin")
    addrow("r_p_kpc",       r_p_chain,           "kpc")
    addrow("R_half_2d_kpc", R_half_2d_chain,     "kpc")
    addrow("r_half_3d_kpc", r_half_3d_chain,     "kpc")
    addrow("alpha_c_deg",   alpha_c_chain / jdf.DEG, "deg")
    addrow("sigma_los_at_R_half_2d", sigma_at_Rhalf, "km/s")
    addrow("log10_M_half_2d", log10_M_2d, "log10(Msun)")
    addrow("log10_M_half_3d", log10_M_3d, "log10(Msun)")
    for tag, v in log10_J.items():
        addrow(f"log10_J_{tag}", v, "log10(GeV^2/cm^5)")
    for tag, v in log10_D.items():
        addrow(f"log10_D_{tag}", v, "log10(GeV/cm^2)")
    summary_rows.append({"quantity": "V_sys_walker",       "unit": "km/s", **cs["V_sys"]})
    summary_rows.append({"quantity": "sigma_los_walker",    "unit": "km/s", **cs["sigma_int"]})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out("summary", "csv"), index=False, float_format="%.6g")
    logp("\n=== Posterior summary (median, q16, q84) ===")
    for r in summary_rows:
        logp(f"  {r['quantity']:<28s} {r['median']:+.4g}  "
              f"[{r['q16']:+.4g}, {r['q84']:+.4g}]  ({r['unit']})")

    # ---- Plots ----
    # Corner of all 7 sampled params (4 halo + 3 nuisances)
    corner_labels = ["V [km/s]", r"$\log_{10} r_s$ [kpc]",
                      r"$\log_{10}\rho_s$", r"$\tilde\beta$",
                      r"$d$ [kpc]", r"$\varepsilon$", r"$r_{1/2}$ [arcmin]"]
    try:
        import corner
        fig = corner.corner(samples_eq,
                             labels=corner_labels,
                             show_titles=True, quantiles=[0.16, 0.5, 0.84])
        fig.savefig(out("corner_halo", "png"), dpi=140)
        plt.close(fig)
    except ImportError:
        n = samples_eq.shape[1]
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.0*ncols, 2.6*nrows))
        for k in range(n):
            ax = axes.flat[k]
            ax.hist(samples_eq[:, k], bins=50, density=True, color="C0", alpha=0.7)
            ax.set_xlabel(corner_labels[k])
        for k in range(n, nrows*ncols):
            axes.flat[k].axis("off")
        fig.tight_layout()
        fig.savefig(out("corner_halo", "png"), dpi=140)
        plt.close(fig)

    # σ_los profile + 1D KDE at R_half. The eyeball data points and R_½ marker
    # are at posterior-median d (R_kpc_med, R_half_2d_med) — same axis as R_grid.
    R_half_2d_med = float(np.median(R_half_2d_chain))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.fill_between(R_grid, sig_lo, sig_hi, alpha=0.3, color="C0",
                     label="68% band")
    ax.plot(R_grid, sig_med, "C0-", lw=2, label="median")
    ax.errorbar(R_kpc_med, np.abs(stars["Vel"] - stars["Vel"].mean()),
                yerr=stars["e_Vel"], fmt=".", ms=3, alpha=0.4, color="k",
                label="|V-<V>| (data, eyeball)")
    ax.set_xscale("log")
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel(r"$\sigma_{\rm los}(R)$ [km/s]")
    ax.axvline(R_half_2d_med, ls="--", color="grey", label=r"$R_{1/2,2D}$ (median)")
    ax.legend(fontsize=8)
    ax.set_title("σ_los profile posterior")

    ax = axes[1]
    s = sigma_at_Rhalf[np.isfinite(sigma_at_Rhalf)]
    ax.hist(s, bins=60, density=True, color="C0", alpha=0.7)
    ax.set_xlabel(r"$\sigma_{\rm los}(R_{1/2,2D})$ [km/s]")
    ax.set_ylabel("posterior density")
    q16, q50, q84 = np.percentile(s, [16, 50, 84])
    ax.axvline(q50, color="k")
    ax.axvline(q16, color="k", ls="--"); ax.axvline(q84, color="k", ls="--")
    ax.set_title(f"σ at R_½: {q50:.2f} +{q84-q50:.2f}/-{q50-q16:.2f} km/s")
    fig.tight_layout()
    fig.savefig(out("posterior_sigma_los", "png"), dpi=140)
    plt.close(fig)

    # M_half KDE
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for arr, lbl, c in [(log10_M_2d, r"$M(R_{1/2,2D})$", "C0"),
                          (log10_M_3d, r"$M(r_{1/2,3D})$", "C3")]:
        ax.hist(arr, bins=60, density=True, alpha=0.55, color=c, label=lbl)
        q16, q50, q84 = np.percentile(arr, [16, 50, 84])
        ax.axvline(q50, color=c, ls="-")
    ax.set_xlabel(r"$\log_{10}(M / M_\odot)$")
    ax.set_ylabel("posterior density")
    ax.legend()
    ax.set_title("Enclosed-mass posteriors")
    fig.tight_layout()
    fig.savefig(out("posterior_M_half", "png"), dpi=140)
    plt.close(fig)

    # Beta KDE
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(btilde_chain, bins=60, density=True, color="C2", alpha=0.7)
    axes[0].set_xlabel(r"$\tilde\beta$")
    axes[0].set_ylabel("posterior density")
    axes[1].hist(beta_chain, bins=60, density=True, color="C2", alpha=0.7,
                  range=(-3, 1))
    axes[1].set_xlabel(r"$\beta$")
    axes[1].set_ylabel("posterior density")
    fig.suptitle("Anisotropy posterior")
    fig.tight_layout()
    fig.savefig(out("posterior_beta", "png"), dpi=140)
    plt.close(fig)

    # J / D
    def _jd_plot(chains, title, fname, unit_lbl):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for tag, arr in chains.items():
            arr = arr[np.isfinite(arr)]
            ax.hist(arr, bins=50, density=True, alpha=0.45, label=tag)
            q50 = np.median(arr)
            ax.axvline(q50, color=plt.gca().lines[-1].get_color() if False
                        else None, ls=":")
        ax.set_xlabel(unit_lbl)
        ax.set_ylabel("posterior density")
        ax.legend(title="θ")
        ax.set_title(title)
        fig.tight_layout()
        # fname is a stem like "posterior_J" — apply the suffix here.
        stem, _, ext = fname.rpartition(".")
        fig.savefig(out(stem, ext), dpi=140)
        plt.close(fig)

    alpha_c_med_deg = float(np.median(alpha_c_chain)) / jdf.DEG
    _jd_plot(log10_J, f"J-factor posteriors (median α_c={alpha_c_med_deg:.3f}°)",
              "posterior_J.png", r"$\log_{10} J\ [{\rm GeV}^2 / {\rm cm}^5]$")
    _jd_plot(log10_D, f"D-factor posteriors (median α_c/2={0.5*alpha_c_med_deg:.3f}°)",
              "posterior_D.png", r"$\log_{10} D\ [{\rm GeV} / {\rm cm}^2]$")

    # σ_los Walker+2006 marginal posterior (proper Jeffreys / Fisher-det prior on σ).
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sg = cs["sigma_grid"]
    pdf = cs["marg_sigma"]
    ax.fill_between(sg, pdf, color="C0", alpha=0.45)
    ax.plot(sg, pdf, "C0-", lw=1.5)
    q16, q50, q84 = cs["sigma_int"]["q16"], cs["sigma_int"]["median"], cs["sigma_int"]["q84"]
    for q, ls in [(q50, "-"), (q16, "--"), (q84, "--")]:
        ax.axvline(q, color="C0", ls=ls, lw=1)
    ax.set_xlim(0, min(15.0, sg[-1]))
    ax.set_xlabel(r"$\sigma_{\rm los}$ [km/s]")
    ax.set_ylabel("posterior density")
    ax.set_title(
        rf"Segue 1: Walker $\sigma = {q50:.2f}^{{+{q84-q50:.2f}}}_{{-{q50-q16:.2f}}}$ km/s"
        f"  (N={len(stars)})"
    )
    fig.tight_layout()
    fig.savefig(out("sigma_los_walker", "png"), dpi=140)
    plt.close(fig)

    if geha_audit is not None:
        import json as _json
        out("selection_audit", "json").write_text(
            _json.dumps(geha_audit, indent=2, default=str) + "\n"
        )

    logp(f"\nTotal wall time: {time.time()-t0:.1f}s")
    out("run", "log").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
