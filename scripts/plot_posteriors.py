"""Posterior diagnostic plots for production runs.

Writes four PNGs per galaxy into ``plots/<lvdb_key>/<prior>/`` (relative
to repo root), refreshed from the latest production run on each
invocation. The ``plots/`` directory always reflects the most recent
results.

  * ``jeans_corner.png`` — corner plot of the four Stage-1 Jeans-model
    parameters: $\\bar V$, $\\log_{10} r_s$, $\\log_{10} \\rho_s$,
    $\\tilde\\beta$ (sampled variable, not the unbounded $\\beta$).
  * ``jd_mhalf.png`` — marginals for $\\log_{10} J(0.5°)$,
    $\\log_{10} D(0.5°)$, $\\log_{10} M(r_{1/2,3D})$, plus the J/D
    chains at the other three reporting angles for context.
  * ``m_J_corner.png`` — joint posterior of $\\log_{10} M(r_{1/2,3D})$
    against $\\log_{10} J(0.5°)$. Uses the saved ``idx_jd`` index to
    align M with the thinned J chain; for older runs without
    ``idx_jd``, J(0.5°) is recomputed on a deterministic subsample so
    the (M, J) pairs are aligned by construction.
  * ``sigma_los_walker.png`` — Walker+2006 constant-σ marginal
    posterior (recomputed on-the-fly from the same per-star catalog
    the production run consumed via ``constant_sigma_inference``).

Usage:
    python scripts/plot_posteriors.py --lvdb-key willman_1
    python scripts/plot_posteriors.py --all

Reads from the canonical ``results/production/<lvdb_key>/<prior>/``
directory (overwritten by each production run). ``--all`` iterates
every staged catalog and writes plots for any galaxy that has a
completed posterior. ``--run-dir`` overrides the path explicitly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

from dwarfjeans.jeans.constant_sigma import constant_sigma_inference
from dwarfjeans.jeans.preprocess import prepare_jeans_input
from dwarfjeans.jeans.selection import SelectionPolicy

sys.path.insert(0, str(HERE))
from run_production import _read_registry_row  # noqa: E402
import plot_config  # noqa: F401,E402  — applies rcParams on import


def _latest_run(lvdb_key: str, prior: str, shmr: str | None = None) -> Path:
    """Return the canonical run dir for (lvdb_key, prior[, shmr]).

    Production results live at the fixed path
    ``results/production/<lvdb_key>/<prior>/`` (overwritten on each run);
    no timestamp/jobid layer. For the per-dwarf SHMR-weighted family
    ``satgen_shmr`` the sub-dir is ``<prior>_<shmr>`` so different SHMR
    choices coexist.
    """
    leaf = f"{prior}_{shmr}" if prior == "satgen_shmr" and shmr else prior
    run_dir = REPO / "results" / "production" / lvdb_key / leaf
    if not (run_dir / "posterior_samples.npz").exists():
        raise FileNotFoundError(
            f"No posterior_samples.npz at {run_dir} — has the production "
            f"run for {lvdb_key} ({leaf}) completed?"
        )
    return run_dir


def _walker_posterior(audit: dict) -> dict:
    """Replay prepare_jeans_input with the run's selection policy and
    return the constant_sigma_inference result dict. The σ_los prior is
    taken from audit['prior_name'] so the Walker plot is consistent with
    the Jeans-side prior of the run."""
    lvdb_key = audit["lvdb_key"]
    sel = audit["selection_policy"]
    prior = audit.get("prior_name", "jeffreys")
    # The σ_los Walker baseline has its own prior namespace
    # ({uniform, loguniform, jeffreys}); the (r_s, ρ_s) `satgen` /
    # `satgen_box` / `satgen_shmr` priors have no σ_los counterpart,
    # so use the production-default `jeffreys` σ_los prior when the
    # Jeans-side run used one of them.
    if prior in ("satgen", "satgen_box", "satgen_shmr"):
        prior = "jeffreys"
    catalog = np.load(REPO / "data" / "star_catalogs" / f"{lvdb_key}.npz",
                      allow_pickle=True)
    registry_row = _read_registry_row(lvdb_key)
    arrays, _ = prepare_jeans_input(
        catalog,
        registry_row,
        selection_policy=SelectionPolicy(
            p_min=float(sel["p_min"]),
            R_over_rhalf_max=float(sel["R_over_rhalf_max"]),
            drop_variable=bool(sel["drop_variable"]),
        ),
    )
    V = arrays["V"]
    sigma_eps = arrays["sigma_eps"]
    p = arrays["p"]
    V_center = float(registry_row.get("vlos_systemic_kms", np.median(V)))
    if np.isnan(V_center):
        V_center = float(np.median(V))
    return constant_sigma_inference(V, sigma_eps, p, V_center=V_center,
                                     prior=prior)


def _q(arr, qs=(0.16, 0.5, 0.84)):
    return np.quantile(arr, qs)


def plot_sigma_walker(walker: dict, lvdb_key: str, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigma = walker["sigma_grid"]
    pdf = walker["marg_sigma"]
    s = walker["sigma_int"]
    q16, q50, q84 = s["q16"], s["median"], s["q84"]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(sigma, pdf, color="C0", lw=1.6)
    ax.axvline(q50, color="C0", ls="--", lw=0.9, label=f"median = {q50:.2f}")
    ax.axvspan(q16, q84, alpha=0.18, color="C0",
               label=f"68% CI = [{q16:.2f}, {q84:.2f}]")
    ax.set_xlim(max(0.0, q16 - 2), q84 + 3)
    ax.set_xlabel(r"$\sigma_\mathrm{los}$ Walker  [km/s]")
    ax.set_ylabel("posterior PDF")
    ax.set_title(rf"{lvdb_key} -- Walker $\sigma_\mathrm{{los}}$ = "rf"{q50:.2f} +{q84 - q50:.2f}/$-${q50 - q16:.2f} km/s")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_jeans_corner(npz, lvdb_key: str, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import corner

    samples = np.column_stack([
        npz["V"],
        npz["log10_rs"],
        npz["log10_rhos"],
        npz["beta_tilde"],
    ])
    labels = [
        r"$\bar V$  [km/s]",
        r"$\log_{10}(r_s / \mathrm{kpc})$",
        r"$\log_{10}(\rho_s / (\mathrm{M}_\odot\,\mathrm{kpc}^{-3}))$",
        r"$\tilde\beta$",
    ]
    import matplotlib.colors as mcolors
    grey = mcolors.to_rgb("tab:grey")
    fill_colors = [grey + (0.0,), grey + (0.5,), grey + (0.8,)]
    fig = plt.figure(figsize=(12, 12))
    corner.corner(
        samples,
        labels=labels,
        quantiles=(0.16, 0.5, 0.84),
        show_titles=True,
        title_fmt=".3g",
        plot_datapoints=False,
        fill_contours=True,
        smooth=0.7,
        smooth1d=0.7,
        levels=(0.68, 0.95),
        color="tab:grey",
        contour_kwargs={"colors": [grey + (0.8,), grey + (0.5,)]},
        contourf_kwargs={"colors": fill_colors},
        hist_kwargs={"color": "black"},
        fig=fig,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_jd_mhalf(npz, lvdb_key: str, out_path: Path) -> Path:
    """3×4 grid. Rows = quantity (J, D, mass / dispersion); columns =
    angle ascending (0.1°, 0.2°, 0.5°, α_c)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layout = [
        # Row 1: J at the four reporting angles, ascending
        [
            ("log10_J_0p1deg",       r"$\log_{10}(J(0.1^{\circ}) / (\mathrm{GeV}^2\,\mathrm{cm}^{-5}))$"),
            ("log10_J_0p2deg",       r"$\log_{10}(J(0.2^{\circ}) / (\mathrm{GeV}^2\,\mathrm{cm}^{-5}))$"),
            ("log10_J_0p5deg",       r"$\log_{10}(J(0.5^{\circ}) / (\mathrm{GeV}^2\,\mathrm{cm}^{-5}))$"),
            ("log10_J_alphac",       r"$\log_{10}(J(\alpha_c) / (\mathrm{GeV}^2\,\mathrm{cm}^{-5}))$"),
        ],
        # Row 2: D at matching angles (α_c/2 in the natural-angle slot)
        [
            ("log10_D_0p1deg",       r"$\log_{10}(D(0.1^{\circ}) / (\mathrm{GeV}\,\mathrm{cm}^{-2}))$"),
            ("log10_D_0p2deg",       r"$\log_{10}(D(0.2^{\circ}) / (\mathrm{GeV}\,\mathrm{cm}^{-2}))$"),
            ("log10_D_0p5deg",       r"$\log_{10}(D(0.5^{\circ}) / (\mathrm{GeV}\,\mathrm{cm}^{-2}))$"),
            ("log10_D_alphacover2",  r"$\log_{10}(D(\alpha_c/2) / (\mathrm{GeV}\,\mathrm{cm}^{-2}))$"),
        ],
        # Row 3: derived masses + projected dispersion at R_½,2D
        [
            ("log10_M_half_2d",      r"$\log_{10}(M(R_{1/2,2D}) / \mathrm{M}_\odot)$"),
            ("log10_M_half_3d",      r"$\log_{10}(M(r_{1/2,3D}) / \mathrm{M}_\odot)$"),
            ("sigma_los_at_Rhalf2d", r"$\sigma_\mathrm{los}(R_{1/2,2D})$ Jeans  [km/s]"),
            (None, None),
        ],
    ]

    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    fig.suptitle(f"{lvdb_key} — J / D / M$_{{1/2}}$ posteriors", fontsize=13)

    for i, row in enumerate(layout):
        for j, (key, xlabel) in enumerate(row):
            ax = axes[i, j]
            if key is None:
                ax.axis("off")
                continue
            x = npz[key]
            q = _q(x)
            ax.hist(x, bins=60, color="C0", alpha=0.6, density=True)
            ax.axvline(q[1], color="k", ls="--", lw=0.9)
            ax.axvspan(q[0], q[2], alpha=0.12, color="k")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("density")
            ax.set_title(rf"{q[1]:.3g} +{q[2]-q[1]:.2g}/$-${q[1]-q[0]:.2g}",
                         fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _aligned_M_J(npz, n_subsample: int = 500):
    """Return aligned (log10 M(r_½,3D), log10 J(0.5°)) sample arrays.

    Uses the saved ``idx_jd`` index when present (production runs
    written after the run_production.py update). Falls back to
    recomputing J(0.5°) on a deterministic ``np.linspace`` subsample
    of the full chain when ``idx_jd`` is absent (older runs)."""
    M = npz["log10_M_half_3d"]
    if "idx_jd" in npz.files:
        idx = npz["idx_jd"]
        return M[idx], npz["log10_J_0p5deg"]

    # Fallback: recompute J(0.5°) for a deterministic subsample.
    from dwarfjeans.jd.factors import J_D_factors, LOG10_J_FAC
    rs = 10.0 ** npz["log10_rs"]
    rhos = 10.0 ** npz["log10_rhos"]
    d = npz["d_kpc_chain"]
    r_t = float(np.median(npz["r_t_kpc"]))
    n = min(n_subsample, len(M))
    idx = np.linspace(0, len(M) - 1, n, dtype=int)
    theta = np.deg2rad(0.5)
    J = np.full(n, np.nan)
    for j, i in enumerate(idx):
        try:
            Jval, _ = J_D_factors(theta, float(d[i]), float(rs[i]),
                                  float(rhos[i]), r_t)
            if Jval > 0:
                J[j] = np.log10(Jval) + LOG10_J_FAC
        except Exception:
            pass
    return M[idx], J


def plot_m_J_corner(npz, lvdb_key: str, out_path: Path) -> Path:
    """2D corner of log10 M(r_½,3D) vs log10 J(0.5°)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import corner

    M, J = _aligned_M_J(npz)
    mask = np.isfinite(M) & np.isfinite(J)
    M, J = M[mask], J[mask]

    samples = np.column_stack([M, J])
    labels = [r"$\log_{10}(M(r_{1/2,3D}) / \mathrm{M}_\odot)$",
              r"$\log_{10}(J(0.5^{\circ}) / (\mathrm{GeV}^2\,\mathrm{cm}^{-5}))$"]
    import matplotlib.colors as mcolors
    grey = mcolors.to_rgb("tab:grey")
    fill_colors = [grey + (0.0,), grey + (0.5,), grey + (0.8,)]
    fig = plt.figure(figsize=(12, 12))
    corner.corner(
        samples,
        labels=labels,
        quantiles=(0.16, 0.5, 0.84),
        show_titles=True,
        title_fmt=".3g",
        plot_datapoints=False,
        fill_contours=True,
        smooth=0.7,
        smooth1d=0.7,
        levels=(0.68, 0.95),
        color="tab:grey",
        contour_kwargs={"colors": [grey + (0.8,), grey + (0.5,)]},
        contourf_kwargs={"colors": fill_colors},
        hist_kwargs={"color": "black"},
        fig=fig,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


PLOTS_DIR = REPO / "plots"


def make_plots(run_dir: Path, out_dir: Path) -> list[Path]:
    npz = np.load(run_dir / "posterior_samples.npz")
    audit = json.loads((run_dir / "audit.json").read_text())
    lvdb_key = audit["lvdb_key"]

    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    out.append(plot_jeans_corner(npz, lvdb_key, out_dir / "jeans_corner.pdf"))
    out.append(plot_jd_mhalf(npz, lvdb_key,    out_dir / "jd_mhalf.pdf"))
    out.append(plot_m_J_corner(npz, lvdb_key,  out_dir / "m_J_corner.pdf"))
    out.append(plot_sigma_walker(_walker_posterior(audit), lvdb_key,
                                  out_dir / "sigma_los_walker.pdf"))
    return out


def _staged_keys() -> list[str]:
    return sorted(p.stem for p in (REPO / "data" / "star_catalogs").glob("*.npz"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--lvdb-key", help="Single galaxy. Mutually exclusive with --all.")
    p.add_argument("--all", action="store_true",
                   help="Iterate every staged catalog with a completed posterior.")
    p.add_argument("--prior", default="jeffreys",
                   choices=("uniform", "loguniform", "jeffreys",
                            "satgen", "satgen_box", "satgen_shmr"))
    p.add_argument("--shmr", default=None,
                   choices=("fattahi18",),
                   help="SHMR for satgen_shmr (required iff --prior satgen_shmr)")
    p.add_argument("--run-dir", default=None,
                   help="Override the auto-discovered latest run dir "
                        "(only valid with --lvdb-key)")
    args = p.parse_args()

    if args.all == bool(args.lvdb_key):
        p.error("specify exactly one of --lvdb-key or --all")
    if args.run_dir and not args.lvdb_key:
        p.error("--run-dir requires --lvdb-key")
    if args.prior == "satgen_shmr" and args.shmr is None:
        p.error("--prior satgen_shmr requires --shmr")
    if args.shmr is not None and args.prior != "satgen_shmr":
        p.error("--shmr is only valid with --prior satgen_shmr")

    if args.lvdb_key:
        keys = [args.lvdb_key]
    else:
        keys = _staged_keys()

    n_done = 0
    for key in keys:
        try:
            run_dir = (Path(args.run_dir).resolve()
                       if args.run_dir else _latest_run(key, args.prior, args.shmr))
        except FileNotFoundError as e:
            print(f"  skip  {key}: {e}")
            continue
        # The plot subdir must reflect the prior of the *run* (read from
        # audit.json), not the CLI flag, otherwise --run-dir pointing at
        # a run whose prior differs from --prior silently overwrites
        # plots in the wrong directory. For satgen_shmr the leaf also
        # encodes the SHMR (matches the results-dir convention), since
        # audit.json stores only prior_name and not the shmr selector.
        audit = json.loads((run_dir / "audit.json").read_text())
        effective_prior = audit.get("prior_name", args.prior)
        effective_shmr = audit.get("shmr") or args.shmr
        leaf = effective_prior
        if effective_prior == "satgen_shmr" and effective_shmr:
            leaf = f"{effective_prior}_{effective_shmr}"
        out_dir = PLOTS_DIR / key / leaf
        for p in make_plots(run_dir, out_dir):
            print(p)
        n_done += 1
    print(f"plots refreshed for {n_done}/{len(keys)} galaxies in {PLOTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
