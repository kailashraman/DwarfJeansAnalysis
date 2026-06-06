"""Regenerate derived views (summary.csv + plots) from saved chain files.

Loops over every ``<base>/<lvdb_key>/<prior>/posterior_samples.npz`` without
re-running dynesty. New chain-only npz files carry their own reproducibility
metadata; legacy npz files (written before the chain-only contract) fall back
to the run's ``audit.json`` for the selection policy, rseed, and thinning, so
the existing results tree can be backfilled in place.

Usage:
    python scripts/reprocess.py
    python scripts/reprocess.py --base results/production --glob '*/*'
    python scripts/reprocess.py --no-plots
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

from dwarfjeans import postprocess as pp
from dwarfjeans.jd import tidal as jdtidal

sys.path.insert(0, str(HERE))
import plot_posteriors  # noqa: E402


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--base", default=str(REPO / "results" / "production"),
                   help="Base directory to scan (default: results/production)")
    p.add_argument("--glob", default="*/*",
                   help="Glob pattern relative to --base for <lvdb_key>/<prior> "
                        "subdirs (default: '*/*')")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot regeneration; write only summary.csv")
    return p.parse_args()


def main() -> int:
    args = _cli()
    base = Path(args.base).resolve()
    if not base.exists():
        print(f"ERROR: base directory does not exist: {base}", file=sys.stderr)
        return 1

    npz_paths = sorted(base.glob(f"{args.glob}/posterior_samples.npz"))
    if not npz_paths:
        print(f"No posterior_samples.npz files found under {base}/{args.glob}")
        return 0

    n_ok = 0
    n_skip = 0
    n_fail = 0
    failures: list[str] = []

    for npz_path in npz_paths:
        run_dir = npz_path.parent
        label = str(run_dir.relative_to(base))
        try:
            samples_eq, meta = pp.load_chain(npz_path)

            origin = "chain"
            if "selection" not in meta or meta.get("lvdb_key") is None:
                legacy = pp.legacy_meta_from_audit(run_dir)
                if legacy is None:
                    print(f"  SKIP  {label}: legacy npz and no usable "
                          "audit.json selection_policy; cannot reprocess")
                    n_skip += 1
                    continue
                meta = legacy
                origin = "audit.json"

            lvdb_key = meta["lvdb_key"]
            prior_name = meta.get("prior_name", "jeffreys")
            ctx = pp.prepare(lvdb_key, prior_name=prior_name, **meta["selection"])
            derived = pp.derive(
                samples_eq, ctx,
                rseed=meta.get("rseed", 0),
                thin_sigma=meta.get("thin_sigma", 2000),
                thin_jd=meta.get("thin_jd", 500),
                thin_profile=meta.get("thin_profile", 300),
                host=meta.get("host", jdtidal.SATGEN_M12_HOST),
                tidal_factor=meta.get("tidal_factor", 2.0),
            )
            pp.write_summary_csv(
                run_dir / "summary.csv",
                pp.summary_rows(derived, ctx),
            )

            if not args.no_plots:
                plots_leaf = run_dir.relative_to(base)
                out_dir = REPO / "plots" / plots_leaf
                # Reuse the ctx/derived just computed — avoid a second derive().
                plot_posteriors.make_plots(run_dir, out_dir, ctx=ctx,
                                           derived=derived, prior_name=prior_name)

            print(f"  OK    {label}  ({origin})")
            n_ok += 1

        except Exception:
            msg = traceback.format_exc().strip().splitlines()[-1]
            print(f"  FAIL  {label}: {msg}")
            failures.append(label)
            n_fail += 1

    total = n_ok + n_skip + n_fail
    print(f"\nDone: {n_ok} reprocessed, {n_skip} skipped, {n_fail} failed "
          f"(of {total} runs found)")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  {f}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
