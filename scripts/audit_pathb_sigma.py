"""Audit Path B σ_los: LVDB-quoted vs constant-σ posterior.

Read-only aggregation across already-completed production runs. Joins:
  - data/registry/galaxies.ecsv (Path B rows)
  - data/lvdb_v1.0.5/comb_all.csv (asymmetric LVDB σ errors)
  - results/production/<key>/{jeffreys,loguniform}/{summary.csv,audit.json}

Writes results/audits/pathb_sigma_comparison.csv and
docs/path_b_sigma_audit.md, sorted by |tension| under the Jeffreys prior.
"""
from __future__ import annotations

import csv
import json
import math
import shlex
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data" / "registry" / "galaxies.ecsv"
LVDB = REPO / "data" / "lvdb_v1.0.5" / "comb_all.csv"
RESULTS = REPO / "results" / "production"
OUT_CSV = REPO / "results" / "audits" / "pathb_sigma_comparison.csv"
OUT_MD = REPO / "docs" / "path_b_sigma_audit.md"

PRIORS = ("jeffreys", "loguniform")

# arXiv preprint IDs for Path B `ref_vlos` papers. Verified by WebSearch +
# WebFetch on 2026-05-12: each URL resolves and title/first author match
# the ADS bibcode.
ARXIV_BY_BIBCODE = {
    "Ji2021ApJ...921...32J":         "2106.12656",
    "Bruce2023ApJ...950..167B":      "2302.03708",
    "Walker2009AJ....137.3100W":     "0811.0118",
    "Heiger2024ApJ...961..234H":     "2308.08602",
    "Chiti2022ApJ...939...41C":      "2206.04580",
    "Tan2025ApJ...979..176T":        "2408.00865",
    "Kirby2015ApJ...810...56K":      "1506.01021",
    "Chiti2023AJ....165...55C":      "2205.01740",
    "Simon2020ApJ...892..137S":      "1911.08493",
    "Hansen2024ApJ...968...21H":     "2403.13060",
    "Li2018ApJ...857..145L":         "1802.06810",
    "Koposov2015ApJ...811...62K":    "1504.07916",
    "Koposov2018MNRAS.479.5343K":    "1804.06430",
    "Walker2015ApJ...808..108W":     "1504.03060",
    "Li2017ApJ...838....8L":         "1611.05052",
}


def _read_registry_pathB():
    rows = []
    header = None
    for line in REGISTRY.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        toks = shlex.split(line)
        if header is None:
            header = toks
            continue
        rec = dict(zip(header, toks))
        if rec.get("path") == "B":
            rows.append(rec)
    return rows


def _read_lvdb_sigma():
    out = {}
    with LVDB.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = r["key"]
            def _f(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return None
            out[key] = {
                "vlos_sigma": _f(r.get("vlos_sigma")),
                "vlos_sigma_em": _f(r.get("vlos_sigma_em")),
                "vlos_sigma_ep": _f(r.get("vlos_sigma_ep")),
                "vlos_sigma_ul": _f(r.get("vlos_sigma_ul")),
                "ref_vlos": r.get("ref_vlos", ""),
            }
    return out


def _read_summary_sigma(path):
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["quantity"] == "sigma_los_walker_kms":
                return float(r["q16"]), float(r["q50"]), float(r["q84"])
    raise KeyError(f"sigma_los_walker_kms not in {path}")


def _read_audit_nfinal(path):
    a = json.loads(path.read_text())
    return int(a["prepare_jeans_input_audit"]["selection"]["n_final"])


def _sym_err(em, ep):
    if em is None and ep is None:
        return None
    if em is None:
        return ep
    if ep is None:
        return em
    return 0.5 * (em + ep)


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    reg = _read_registry_pathB()
    lvdb = _read_lvdb_sigma()

    rows = []
    for r in reg:
        key = r["lvdb_key"]
        ref_vlos = r.get("ref_vlos", "")
        l = lvdb.get(key, {})
        s_lvdb = l.get("vlos_sigma")
        em_lvdb = l.get("vlos_sigma_em")
        ep_lvdb = l.get("vlos_sigma_ep")
        ul_lvdb = l.get("vlos_sigma_ul")
        is_ul = (ul_lvdb is not None) and (not math.isnan(ul_lvdb))
        unresolved_flag = str(r.get("vlos_sigma_unresolved", "")).lower() == "true"

        per_prior = {}
        n_final = None
        for prior in PRIORS:
            sdir = RESULTS / key / prior
            spath = sdir / "summary.csv"
            apath = sdir / "audit.json"
            if not spath.exists():
                per_prior[prior] = None
                continue
            q16, q50, q84 = _read_summary_sigma(spath)
            per_prior[prior] = (q16, q50, q84)
            if n_final is None and apath.exists():
                try:
                    n_final = _read_audit_nfinal(apath)
                except Exception:
                    pass

        j = per_prior.get("jeffreys")
        lg = per_prior.get("loguniform")

        # Tension z-scores (symmetrized errors). NaN if LVDB σ missing or UL.
        def _tension(post):
            if post is None or s_lvdb is None or is_ul:
                return None
            q16, q50, q84 = post
            err_post = 0.5 * (q84 - q16)
            err_lvdb = _sym_err(em_lvdb, ep_lvdb)
            if err_lvdb is None or err_lvdb <= 0:
                return None
            denom = math.sqrt(err_post ** 2 + err_lvdb ** 2)
            if denom <= 0:
                return None
            return (q50 - s_lvdb) / denom

        z_j = _tension(j)
        z_lg = _tension(lg)
        dsig_jeff = (j[1] - s_lvdb) if (j is not None and s_lvdb is not None) else None
        dsig_prior = (j[1] - lg[1]) if (j is not None and lg is not None) else None

        arxiv_id = ARXIV_BY_BIBCODE.get(ref_vlos, "")
        ref_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
        rows.append({
            "lvdb_key": key,
            "ref_vlos": ref_vlos,
            "ref_url": ref_url,
            "n_final": n_final,
            "sigma_lvdb": s_lvdb,
            "sigma_lvdb_em": em_lvdb,
            "sigma_lvdb_ep": ep_lvdb,
            "lvdb_unresolved_or_UL": is_ul or unresolved_flag,
            "sigma_jeff_q16": j[0] if j else None,
            "sigma_jeff_q50": j[1] if j else None,
            "sigma_jeff_q84": j[2] if j else None,
            "sigma_loguni_q16": lg[0] if lg else None,
            "sigma_loguni_q50": lg[1] if lg else None,
            "sigma_loguni_q84": lg[2] if lg else None,
            "dsigma_jeff_minus_lvdb": dsig_jeff,
            "dsigma_jeff_minus_loguni": dsig_prior,
            "tension_jeff_sigma": z_j,
            "tension_loguni_sigma": z_lg,
        })

    rows.sort(key=lambda r: abs(r["tension_jeff_sigma"]) if r["tension_jeff_sigma"] is not None else -1,
              reverse=True)

    # ---- CSV ----
    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- Markdown ----
    def _fmt(x, prec=2):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{x:.{prec}f}"

    md = ["# Path B σ_los audit: LVDB-quoted vs constant-σ posterior",
          "",
          ("Comparison of the LVDB-quoted line-of-sight velocity dispersion "
           "against the constant-σ (Walker+2006-style, radius-independent) "
           "posterior recovered by our pipeline under both Jeffreys and "
           "loguniform priors. Tension z-score uses symmetrized errors: "
           "`(σ_const − σ_LVDB) / sqrt((Δσ_post/2)² + ((em+ep)/2)²)`. "
           "Rows with LVDB upper limits are flagged and z-score is blank."),
          "",
          ("Sorted by |tension| under the Jeffreys prior. Generated by "
           "`scripts/audit_pathb_sigma.py`. The `ref_vlos` column links to "
           "the paper's arXiv preprint."),
          "",
          ("| lvdb_key | ref_vlos | N | σ_LVDB (+ep/−em) | UL? | "
           "σ_jeff (+ep/−em) | σ_loguni (+ep/−em) | "
           "Δσ (jeff−LVDB) | Δσ (jeff−loguni) | z (jeff) | z (loguni) |"),
          ("|---|---|---:|---|:--:|---|---|---:|---:|---:|---:|")]

    for r in rows:
        if r["sigma_lvdb"] is None:
            sig_lvdb_str = "—"
        else:
            em = r["sigma_lvdb_em"]; ep = r["sigma_lvdb_ep"]
            em_s = _fmt(em) if em is not None else "—"
            ep_s = _fmt(ep) if ep is not None else "—"
            sig_lvdb_str = f"{_fmt(r['sigma_lvdb'])} (+{ep_s}/−{em_s})"
        def _p(q16, q50, q84):
            if q50 is None:
                return "—"
            return f"{_fmt(q50)} (+{_fmt(q84 - q50)}/−{_fmt(q50 - q16)})"
        if r["ref_vlos"] and r["ref_url"]:
            ref_md = f"[{r['ref_vlos']}]({r['ref_url']})"
        elif r["ref_vlos"]:
            ref_md = r["ref_vlos"]
        else:
            ref_md = "—"
        md.append(
            f"| {r['lvdb_key']} | {ref_md} | "
            f"{r['n_final'] if r['n_final'] is not None else '—'} | "
            f"{sig_lvdb_str} | "
            f"{'Y' if r['lvdb_unresolved_or_UL'] else ''} | "
            f"{_p(r['sigma_jeff_q16'], r['sigma_jeff_q50'], r['sigma_jeff_q84'])} | "
            f"{_p(r['sigma_loguni_q16'], r['sigma_loguni_q50'], r['sigma_loguni_q84'])} | "
            f"{_fmt(r['dsigma_jeff_minus_lvdb'])} | "
            f"{_fmt(r['dsigma_jeff_minus_loguni'])} | "
            f"{_fmt(r['tension_jeff_sigma'])} | "
            f"{_fmt(r['tension_loguni_sigma'])} |"
        )

    # tension summary
    z_vals = [r["tension_jeff_sigma"] for r in rows
              if r["tension_jeff_sigma"] is not None]
    n_total = len(rows)
    n_z = len(z_vals)
    n1 = sum(1 for z in z_vals if abs(z) > 1)
    n2 = sum(1 for z in z_vals if abs(z) > 2)
    n3 = sum(1 for z in z_vals if abs(z) > 3)
    md += ["",
           "## Summary",
           "",
           f"- Path B dwarfs in registry: **{n_total}**",
           f"- With computable tension z-score (Jeffreys, LVDB σ resolved): **{n_z}**",
           f"- |z| > 1: **{n1}**;  |z| > 2: **{n2}**;  |z| > 3: **{n3}**"]

    OUT_MD.write_text("\n".join(md) + "\n")

    print(f"wrote {OUT_CSV.relative_to(REPO)}")
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    print(f"Path B total: {n_total};  |z|>1: {n1};  |z|>2: {n2};  |z|>3: {n3}")


if __name__ == "__main__":
    main()
