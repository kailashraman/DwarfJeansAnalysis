"""Posterior diagnostic plots for production runs.

Writes four PNGs per galaxy into ``plots/<lvdb_key>/`` (relative to repo
root), refreshed from the latest production run on each invocation. The
``plots/`` directory always reflects the most recent results.

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
    python scripts/plot_posteriors.py --lvdb-key willman_1 \\
        --run-dir results/production/_slurm_22439950/_runs/willman_1/jeffreys/<ts>

If ``--run-dir`` is omitted the latest run is auto-discovered. ``--all``
iterates every staged catalog and writes plots for any galaxy that has
at least one completed posterior.
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


def _latest_run(lvdb_key: str, prior: str) -> Path:
    candidates = sorted(
        REPO.glob(f"results/production/**/_runs/{lvdb_key}/{prior}/*"),
        key=lambda p: p.name,
    ) + sorted(
        REPO.glob(f"results/production/{lvdb_key}/{prior}/*"),
        key=lambda p: p.name,
    )
    candidates = [c for c in candidates if (c / "posterior_samples.npz").exists()]
    if not candidates:
        raise FileNotFoundError(
            f"No posterior_samples.npz under results/production for "
            f"{lvdb_key}/{prior}"
        )
    return candidates[-1]


def _walker_posterior(audit: dict) -> dict:
    """Replay prepare_jeans_input with the run's selection policy and
    return the constant_sigma_inference result dict."""
    lvdb_key = audit["lvdb_key"]
    sel = audit["selection_policy"]
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
    return constant_sigma_inference(V, sigma_eps, p, V_center=V_center)


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
    ax.set_xlabel("σ_los Walker  [km/s]")
    ax.set_ylabel("posterior PDF")
    ax.set_title(f"{lvdb_key} — Walker σ_los = "
                 f"{q50:.2f} +{q84 - q50:.2f}/−{q50 - q16:.2f} km/s")
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
        r"$\log_{10}\,r_s$  [kpc]",
        r"$\log_{10}\,\rho_s$  [M$_\odot$ kpc$^{-3}$]",
        r"$\tilde\beta$",
    ]
    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=(0.16, 0.5, 0.84),
        show_titles=True,
        title_fmt=".3g",
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 11},
        plot_datapoints=False,
        fill_contours=True,
        levels=(0.39, 0.86),  # 1σ / 2σ in 2D
        color="C0",
    )
    fig.suptitle(f"{lvdb_key} — Jeans posterior", fontsize=12, y=1.0)
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
            ("log10_J_0p1deg",       r"$\log_{10} J(0.1°)$  [GeV$^2$ cm$^{-5}$]"),
            ("log10_J_0p2deg",       r"$\log_{10} J(0.2°)$"),
            ("log10_J_0p5deg",       r"$\log_{10} J(0.5°)$"),
            ("log10_J_alphac",       r"$\log_{10} J(\alpha_c)$"),
        ],
        # Row 2: D at matching angles (α_c/2 in the natural-angle slot)
        [
            ("log10_D_0p1deg",       r"$\log_{10} D(0.1°)$  [GeV cm$^{-2}$]"),
            ("log10_D_0p2deg",       r"$\log_{10} D(0.2°)$"),
            ("log10_D_0p5deg",       r"$\log_{10} D(0.5°)$"),
            ("log10_D_alphacover2",  r"$\log_{10} D(\alpha_c/2)$"),
        ],
        # Row 3: derived masses + projected dispersion at R_½,2D
        [
            ("log10_M_half_2d",      r"$\log_{10} M(R_{1/2,2D})$  [M$_\odot$]"),
            ("log10_M_half_3d",      r"$\log_{10} M(r_{1/2,3D})$  [M$_\odot$]"),
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
            ax.set_title(f"{q[1]:.3g} +{q[2]-q[1]:.2g}/−{q[1]-q[0]:.2g}",
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
    labels = [r"$\log_{10} M(r_{1/2,3D})$  [M$_\odot$]",
              r"$\log_{10} J(0.5°)$  [GeV$^2$ cm$^{-5}$]"]
    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=(0.16, 0.5, 0.84),
        show_titles=True,
        title_fmt=".3g",
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 11},
        plot_datapoints=False,
        fill_contours=True,
        levels=(0.39, 0.86),
        color="C0",
    )
    fig.suptitle(f"{lvdb_key} — M(r$_{{1/2,3D}}$) vs J(0.5°)  (N={len(M)})",
                 fontsize=12, y=1.0)
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
    out.append(plot_jeans_corner(npz, lvdb_key, out_dir / "jeans_corner.png"))
    out.append(plot_jd_mhalf(npz, lvdb_key,    out_dir / "jd_mhalf.png"))
    out.append(plot_m_J_corner(npz, lvdb_key,  out_dir / "m_J_corner.png"))
    out.append(plot_sigma_walker(_walker_posterior(audit), lvdb_key,
                                  out_dir / "sigma_los_walker.png"))
    return out


def _staged_keys() -> list[str]:
    return sorted(p.stem for p in (REPO / "data" / "star_catalogs").glob("*.npz"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--lvdb-key", help="Single galaxy. Mutually exclusive with --all.")
    p.add_argument("--all", action="store_true",
                   help="Iterate every staged catalog with a completed posterior.")
    p.add_argument("--prior", default="jeffreys",
                   choices=("uniform", "loguniform", "jeffreys"))
    p.add_argument("--run-dir", default=None,
                   help="Override the auto-discovered latest run dir "
                        "(only valid with --lvdb-key)")
    args = p.parse_args()

    if args.all == bool(args.lvdb_key):
        p.error("specify exactly one of --lvdb-key or --all")
    if args.run_dir and not args.lvdb_key:
        p.error("--run-dir requires --lvdb-key")

    if args.lvdb_key:
        keys = [args.lvdb_key]
    else:
        keys = _staged_keys()

    n_done = 0
    for key in keys:
        try:
            run_dir = (Path(args.run_dir).resolve()
                       if args.run_dir else _latest_run(key, args.prior))
        except FileNotFoundError as e:
            print(f"  skip  {key}: {e}")
            continue
        out_dir = PLOTS_DIR / key
        for p in make_plots(run_dir, out_dir):
            print(p)
        n_done += 1
    print(f"plots refreshed for {n_done}/{len(keys)} galaxies in {PLOTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
