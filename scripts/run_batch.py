"""Batch driver: run scripts/run_production.py over every staged catalog.

Iterates the lvdb keys present in ``data/star_catalogs/*.npz``, invokes
``run_production.py`` as a subprocess per galaxy, and writes a top-level
``results/production/_batch_<UTC>/manifest.csv`` summarising status,
wall-clock, and the per-galaxy output directory.

Per-galaxy failures are logged and skipped (the batch keeps going);
stderr/stdout for failures lands in ``<batch_dir>/<lvdb_key>.log``.

Usage:
    python scripts/run_batch.py [--prior jeffreys] [--nlive 500] [--dlogz 0.1]
                                [--only key1,key2] [--skip key3]
                                [--jobs 1]

``--jobs N`` runs N galaxies in parallel (each dynesty call is itself
single-threaded by default, so this scales linearly up to core count).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUN_PROD = HERE / "run_production.py"
CATALOG_DIR = REPO / "data" / "star_catalogs"


def _list_keys() -> list[str]:
    return sorted(p.stem for p in CATALOG_DIR.glob("*.npz"))


def _run_one(lvdb_key: str, batch_dir: Path, common_args: list[str]) -> dict:
    log_path = batch_dir / f"{lvdb_key}.log"
    cmd = [sys.executable, str(RUN_PROD), "--lvdb-key", lvdb_key,
           "--output-base", str(batch_dir / "_runs")] + common_args
    t0 = time.time()
    with log_path.open("w") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=str(REPO))
    dt = time.time() - t0
    out_dir = ""
    if proc.returncode == 0:
        # run_production prints the output directory on the last line
        try:
            tail = log_path.read_text().splitlines()
            out_dir = tail[-1].strip() if tail else ""
        except Exception:
            pass
    return {
        "lvdb_key": lvdb_key,
        "returncode": proc.returncode,
        "wall_seconds": round(dt, 1),
        "out_dir": out_dir,
        "log": str(log_path),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--prior", default="jeffreys",
                   choices=("uniform", "loguniform", "jeffreys", "satgen", "satgen_box"))
    p.add_argument("--nlive", type=int, default=500)
    p.add_argument("--dlogz", type=float, default=0.1)
    p.add_argument("--only", default=None,
                   help="Comma-separated lvdb keys to include (default: all)")
    p.add_argument("--skip", default=None,
                   help="Comma-separated lvdb keys to exclude")
    p.add_argument("--jobs", type=int, default=1,
                   help="Galaxies to run in parallel (default 1)")
    args = p.parse_args()

    keys = _list_keys()
    if args.only:
        wanted = set(args.only.split(","))
        keys = [k for k in keys if k in wanted]
    if args.skip:
        drop = set(args.skip.split(","))
        keys = [k for k in keys if k not in drop]
    if not keys:
        print("No galaxies to run.", file=sys.stderr)
        return 1

    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_dir = REPO / "results" / "production" / f"_batch_{ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"Batch dir: {batch_dir}")
    print(f"Galaxies ({len(keys)}): {keys}")

    common = ["--prior", args.prior, "--nlive", str(args.nlive),
              "--dlogz", str(args.dlogz)]

    results: list[dict] = []
    t_batch = time.time()
    if args.jobs <= 1:
        for k in keys:
            print(f"[{k}] starting", flush=True)
            r = _run_one(k, batch_dir, common)
            print(f"[{k}] done rc={r['returncode']} t={r['wall_seconds']}s",
                  flush=True)
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(_run_one, k, batch_dir, common): k
                       for k in keys}
            for fut in as_completed(futures):
                r = fut.result()
                print(f"[{r['lvdb_key']}] done rc={r['returncode']} "
                      f"t={r['wall_seconds']}s", flush=True)
                results.append(r)

    manifest = batch_dir / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["lvdb_key", "returncode",
                                           "wall_seconds", "out_dir", "log"])
        w.writeheader()
        for r in sorted(results, key=lambda x: x["lvdb_key"]):
            w.writerow(r)

    n_ok = sum(1 for r in results if r["returncode"] == 0)
    print(f"\nWrote manifest {manifest}")
    print(f"OK: {n_ok}/{len(results)}  total {time.time()-t_batch:.1f}s")
    return 0 if n_ok == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
