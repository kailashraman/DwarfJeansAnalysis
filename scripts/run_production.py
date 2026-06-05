"""Single-galaxy production driver.

Given an LVDB key, this script runs the full Stage-1 inference:

  1. Load the staged per-star or per-epoch catalog from
     ``data/star_catalogs/<lvdb_key>.npz`` and route it through
     ``prepare_jeans_input`` (which dispatches to the per-paper
     combiner via ``_meta["source_paper_bibcode"]`` and applies
     ``SelectionPolicy``).
  2. Build the 7D nuisance-marginalized galaxy dict (V, sigma_eps, p,
     R, Rad_arcmin) plus nuisance priors derived directly from
     ``data/registry/galaxies.ecsv`` (the registry is the single
     source of truth for distance, rhalf, ellipticity and their
     uncertainties).
  3. Run dynesty with the requested base prior (default: Jeffreys
     conditional on (ln ρ_s, ln r_s)).
  4. Derive per-sample chains for σ_los(R_½,2D), M(R_½,2D),
     M(r_½,3D), J(α_c) + J(0.1°/0.2°/0.5°), D(α_c/2) + D at the same
     fixed angles.
  5. Dump samples + a summary CSV + the selection/combine audit JSON
     to ``results/production/<lvdb_key>/<prior>/`` (overwrites each run).

Usage:
    python scripts/run_production.py --lvdb-key tucana_2

Run ``--help`` for the full argparse interface.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

from dwarfjeans.jeans import inference as jeans_inference
from dwarfjeans.jd import tidal as jdtidal
from dwarfjeans import postprocess as pp

# Backward-compat aliases so external importers (plot_posteriors.py,
# tests/integration/run_segue1.py) keep working without changes.
_read_registry_row = pp.read_registry_row
_registry_nuisance_priors = pp.registry_nuisance_priors


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------

def run(lvdb_key: str,
        *,
        prior_name: str = "jeffreys",
        shmr: str | None = None,
        nlive: int = 500,
        dlogz: float = 0.1,
        rseed: int = 0,
        p_min: float = 0.5,
        rmax_over_rhalf: float = 2.0,
        drop_variable: bool = True,
        use_p_weights: bool = False,
        thin_sigma: int = 2000,
        thin_jd: int = 500,
        thin_profile: int = 300,
        output_base: Path | None = None,
        npool: int = 1,
        ) -> Path:
    """Run the full pipeline for one galaxy. Returns the output dir."""
    if output_base is None:
        output_base = REPO / "results" / "production"
    # Canonical, single-output-per-(galaxy, prior). Each run overwrites
    # the previous one — wrong results are not preserved for provenance,
    # to keep the central results tree from ballooning over re-runs.
    # satgen_shmr is sub-folded by SHMR so different SHMR choices coexist.
    if prior_name == "satgen_shmr":
        if shmr is None:
            raise ValueError("prior_name='satgen_shmr' requires --shmr")
        out_dir = output_base / lvdb_key / f"{prior_name}_{shmr}"
    else:
        out_dir = output_base / lvdb_key / prior_name
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    log: list[str] = []
    t_total0 = time.time()

    def logp(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.append(msg)

    logp(f"=== Production run: {lvdb_key} | prior={prior_name} ===")
    logp(f"out: {out_dir}")

    ctx = pp.prepare(lvdb_key, prior_name=prior_name, p_min=p_min,
                     rmax_over_rhalf=rmax_over_rhalf, drop_variable=drop_variable,
                     use_p_weights=use_p_weights, logp=logp)

    # Log source_paper / nuisance-priors lines that prepare() doesn't emit.
    row = ctx.row
    nuisance_priors = ctx.nuisance_priors
    logp(f"  source_paper:    vlos={row.get('ref_vlos', '?')}, "
         f"struct={row.get('ref_structure', '?')}")
    logp(f"  nuisance priors:")
    for k in ("d_mean", "d_sigma", "eps_mean", "eps_sigma",
              "rhalf_mean", "rhalf_sigma"):
        logp(f"    {k}: {nuisance_priors[k]:.4f}")
    if ctx.selection_audit.get("combine"):
        cd = ctx.selection_audit["combine"]
        logp(f"  combine: n_input_rows={cd.get('n_input_rows')}, "
             f"n_stars={cd.get('n_stars')}, n_variable={cd.get('n_variable')}, "
             f"sigma_sys={cd.get('sigma_sys_kms')}, "
             f"treatment={cd.get('sigma_sys_treatment', 'as_statistical')}, "
             f"offsets={cd.get('zero_point_offsets_kms', {})}")

    ndim = 9 if ctx.perspective_kwargs else 7
    logp(f"\n=== dynesty ({ndim}D, prior={prior_name}, nlive={nlive}, "
         f"dlogz={dlogz}, npool={npool}) ===")
    t_inf = time.time()
    result = jeans_inference.run_inference(
        ctx.galaxy, V_center=ctx.V_center, V_halfwidth=ctx.V_halfwidth,
        nlive=nlive, dlogz=dlogz, rseed=rseed, print_progress=False,
        marginalize_nuisances=True, nuisance_priors=ctx.nuisance_priors,
        prior_name=prior_name, shmr=shmr, lvdb_key=lvdb_key, npool=npool,
        **ctx.perspective_kwargs,
    )
    dt_inf = time.time() - t_inf
    logp(f"  done in {dt_inf:.1f}s")
    logp(f"  logZ = {result['logz']:.3f} ± {result['logz_err']:.3f}")
    logp(f"  n_eq = {result['n_eq']}")

    derived = pp.derive(result["samples_eq"], ctx, rseed=rseed,
                        thin_sigma=thin_sigma, thin_jd=thin_jd,
                        thin_profile=thin_profile, logp=logp)

    pp.save_chain(out_dir / "posterior_samples.npz", result["samples_eq"], ctx,
                  rseed=rseed, prior_name=prior_name, shmr=shmr,
                  thin_sigma=thin_sigma, thin_jd=thin_jd, thin_profile=thin_profile,
                  p_min=p_min, rmax_over_rhalf=rmax_over_rhalf,
                  drop_variable=drop_variable, use_p_weights=use_p_weights)

    pp.write_summary_csv(out_dir / "summary.csv", pp.summary_rows(derived, ctx))

    audit_payload = {
        "lvdb_key": lvdb_key,
        "prior_name": prior_name,
        "shmr": shmr,
        "timestamp_utc": timestamp,
        "registry_row": {k: (v if not isinstance(v, float)
                              or np.isfinite(v) else None)
                          for k, v in ctx.row.items()},
        "selection_policy": {
            "p_min": p_min,
            "R_over_rhalf_max": rmax_over_rhalf,
            "drop_variable": drop_variable,
        },
        "nuisance_priors": ctx.nuisance_priors,
        "prepare_jeans_input_audit": ctx.selection_audit,
        "dynesty": {
            "nlive": nlive, "dlogz": dlogz, "rseed": rseed,
            "logZ": float(result["logz"]),
            "logZ_err": float(result["logz_err"]),
            "n_eq": int(result["n_eq"]),
            "wallclock_s": dt_inf,
        },
        "thinning": {"sigma": derived["n_sigma"],
                      "profile": derived["n_profile"],
                      "jd": derived["n_jd"]},
        "tidal_radius": {
            "convention": "Tormen1998/Springel2008 (factor 2 - dlnM/dlnr)",
            "host": "SatGen m12 NFW",
            "host_Mvir_Msun": jdtidal.SATGEN_M12_HOST.Mvir_Msun,
            "host_Rvir_kpc": jdtidal.SATGEN_M12_HOST.Rvir_kpc,
            "host_concentration": jdtidal.SATGEN_M12_HOST.concentration,
            "r_t_median_kpc": float(np.nanmedian(derived["r_t_chain"])),
        },
    }
    (out_dir / "audit.json").write_text(json.dumps(audit_payload,
                                                    indent=2,
                                                    default=str))

    (out_dir / "run.log").write_text("\n".join(log) + "\n")
    logp(f"\n=== Wrote outputs to {out_dir} (total {time.time()-t_total0:.1f}s) ===")
    return out_dir


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--lvdb-key", required=True,
                   help="Galaxy key in data/registry/galaxies.ecsv")
    p.add_argument("--prior", default="jeffreys",
                   choices=("uniform", "loguniform", "jeffreys",
                            "satgen", "satgen_box", "satgen_shmr"),
                   help="Base halo prior on (ln ρ_s, ln r_s)")
    p.add_argument("--shmr", default=None,
                   choices=("fattahi18", "moster18", "danieli23_const", "kim24"),
                   help="SHMR for satgen_shmr (required iff --prior satgen_shmr)")
    p.add_argument("--nlive", type=int, default=500)
    p.add_argument("--dlogz", type=float, default=0.1)
    p.add_argument("--rseed", type=int, default=0)
    p.add_argument("--p-min", type=float, default=0.5)
    p.add_argument("--rmax-over-rhalf", type=float, default=2.0)
    p.add_argument("--use-p-weights", action="store_true",
                   help="Propagate continuous p_i into the likelihood instead of "
                        "replacing post-cut survivors with p=1 (default).")
    p.add_argument("--keep-variable", action="store_true",
                   help="Disable the variability/χ² drop in selection")
    p.add_argument("--thin-sigma", type=int, default=2000,
                   help="Posterior thin for σ_los at R_½,2D")
    p.add_argument("--thin-profile", type=int, default=300,
                   help="Posterior thin for σ_los radial profile bands")
    p.add_argument("--thin-jd", type=int, default=500,
                   help="Posterior thin for J/D integrals")
    p.add_argument("--output-base", default=None,
                   help="Override results/production")
    p.add_argument("--npool", type=int, default=1,
                   help="Multiprocessing pool size for dynesty likelihood "
                        "evaluations (default 1 = serial). Set to match "
                        "--cpus-per-task on SLURM. Pool order is non-deterministic, "
                        "so posterior medians shift at the ~1%% level vs --npool 1.")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    if args.prior == "satgen_shmr" and args.shmr is None:
        raise SystemExit("--prior satgen_shmr requires --shmr")
    if args.shmr is not None and args.prior != "satgen_shmr":
        raise SystemExit("--shmr is only valid with --prior satgen_shmr")
    out = run(
        args.lvdb_key,
        prior_name=args.prior,
        shmr=args.shmr,
        nlive=args.nlive,
        dlogz=args.dlogz,
        rseed=args.rseed,
        p_min=args.p_min,
        rmax_over_rhalf=args.rmax_over_rhalf,
        drop_variable=not args.keep_variable,
        use_p_weights=args.use_p_weights,
        thin_sigma=args.thin_sigma,
        thin_profile=args.thin_profile,
        thin_jd=args.thin_jd,
        output_base=Path(args.output_base) if args.output_base else None,
        npool=args.npool,
    )
    print(out)
