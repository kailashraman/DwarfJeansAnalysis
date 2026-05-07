"""Stage 0b Path B — emit a worklist of `ref_vlos` papers to stage manually.

Implements docs/plan/data_sources.md §"Path B — LVDB `ref_vlos` paper ingest"
step 1 (resolution) and §4 (staging) by enumerating the work, not by fetching.

VizieR / journal-MRT downloading is intentionally out of scope here — each Path B
paper is its own per-paper engineering task (different schemas, different
membership conventions, different per-epoch / per-star granularity).

Run:
    python -m data_ingest.stage0b_pathb_worklist

Outputs:
    data/path_b_worklist.csv
    data/path_b_worklist.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from astropy.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_ECSV = REPO_ROOT / "data" / "registry" / "galaxies.ecsv"
WORKLIST_CSV = REPO_ROOT / "data" / "path_b_worklist.csv"
WORKLIST_MD = REPO_ROOT / "data" / "path_b_worklist.md"

BIBCODE_LEN = 19


def _split_ref_vlos(ref: str) -> tuple[str, str, str]:
    """Return (last_name, ads_bibcode, year). LVDB convention: '<LastName><19-char ADS>'."""
    if len(ref) < BIBCODE_LEN:
        raise ValueError(f"ref_vlos too short to contain a bibcode: {ref!r}")
    bibcode = ref[-BIBCODE_LEN:]
    last_name = ref[:-BIBCODE_LEN]
    # ADS bibcode YYYY...; first 4 chars are the year.
    year_match = re.match(r"^(\d{4})", bibcode)
    year = year_match.group(1) if year_match else "????"
    return last_name, bibcode, year


def _suggest_bibkey(last_name: str, year: str) -> str:
    return (last_name + year).lower().replace(" ", "").replace(".", "")


def main() -> int:
    reg = Table.read(REGISTRY_ECSV, format="ascii.ecsv")
    rows = []
    for r in reg[reg["path"] == "B"]:
        ref = str(r["ref_vlos"])
        last_name, bibcode, year = _split_ref_vlos(ref)
        bibkey = _suggest_bibkey(last_name, year)
        rows.append({
            "study_name": str(r["study_name"]),
            "lvdb_key": str(r["lvdb_key"]),
            "ra_deg": float(r["ra_deg"]),
            "dec_deg": float(r["dec_deg"]),
            "ref_vlos": ref,
            "ads_bibcode": bibcode,
            "first_author": last_name,
            "year": year,
            "suggested_bibkey_folder": f"data/{bibkey}/",
        })

    df = pd.DataFrame(rows)
    df.to_csv(WORKLIST_CSV, index=False)

    # Group by paper so the operator stages one folder per shared paper.
    by_paper: dict[str, list[dict]] = {}
    for row in rows:
        by_paper.setdefault(row["ads_bibcode"], []).append(row)

    lines = []
    lines.append("# Path B — per-paper staging worklist")
    lines.append("")
    lines.append(
        f"Generated from `data/registry/galaxies.ecsv` — {len(rows)} Path B galaxies "
        f"across {len(by_paper)} distinct `ref_vlos` papers."
    )
    lines.append("")
    lines.append(
        "For each entry below, follow `docs/plan/data_sources.md` §\"Path B — LVDB "
        "`ref_vlos` paper ingest\" steps 3–5: locate the per-star catalog (VizieR "
        "preferred, journal MRT next, author page last resort), download into the "
        "suggested `data/<bibkey>/` folder, write `PROVENANCE.md`, and run "
        "`sha256sum *.* > checksums.sha256`."
    )
    lines.append("")
    lines.append(
        "After staging, write a per-paper adapter (a small dict / YAML mapping the "
        "paper's column names to canonical `V`, `sigma_eps`, `p`, `star_id`, "
        "`RA_star`, `Dec_star`) and run a per-galaxy ingest analogous to "
        "`data_ingest/stage0b_geha.py:ingest_one`."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    for bibcode, entries in sorted(by_paper.items()):
        first = entries[0]
        lines.append(f"## {first['first_author']} {first['year']} — `{bibcode}`")
        lines.append("")
        lines.append(f"- Suggested folder: `{first['suggested_bibkey_folder']}`")
        lines.append(f"- ADS link: https://ui.adsabs.harvard.edu/abs/{bibcode}")
        lines.append(f"- Galaxies served by this paper:")
        for e in entries:
            lines.append(
                f"  - **{e['study_name']}** (`{e['lvdb_key']}`) — center "
                f"(RA, Dec) = ({e['ra_deg']:.4f}°, {e['dec_deg']:.4f}°)"
            )
        lines.append("")

    WORKLIST_MD.write_text("\n".join(lines))
    print(f"Wrote {WORKLIST_CSV.relative_to(REPO_ROOT)}")
    print(f"Wrote {WORKLIST_MD.relative_to(REPO_ROOT)}")
    print(f"  {len(rows)} galaxies across {len(by_paper)} unique papers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
