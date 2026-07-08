#!/bin/bash
#SBATCH --job-name=djreproc
#SBATCH --time=0:30:00
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr7
#SBATCH --qos=lr_normal
#SBATCH --array=0-38
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=NONE
#SBATCH --exclude=n0150.lr7
#SBATCH --output=/global/scratch/projects/pc_heptheory/kraman/DwarfJeansAnalysis/logs/reproc-%A_%a.out

# Regenerate the derived views (summary.csv + derived.npz + plots) for every existing
# production run WITHOUT re-running dynesty. One array task per galaxy; each
# task reprocesses all of that galaxy's priors via scripts/reprocess.py
# --glob "<key>/*". Derived quantities are recomputed from the saved chain
# (and backfilled from audit.json for legacy npz), so this is the cheap way to
# roll the new tidal-radius / theta95 quantities onto the whole results tree.
#
# Cost: ~8 priors x ~50 s/run ~= 7 min per task; 39 tasks run in parallel.
#
# Usage (launch with `bash` so the shared bad-node exclude list is applied;
# any extra args are forwarded to sbatch):
#   bash scripts/submit_reprocess.sh                   # full 39-galaxy sweep
#   bash scripts/submit_reprocess.sh --array=0         # one galaxy (index 0)
#   NO_PLOTS=1 bash scripts/submit_reprocess.sh        # summary.csv + derived.npz only
#   HOST=mw2022 bash scripts/submit_reprocess.sh       # roll the real-MW tidal host onto every run
#
# A bare `sbatch scripts/submit_reprocess.sh` still works but only honors the
# static #SBATCH --exclude (n0150) fallback above, not the full shared list.
#
# Array indices match the sorted data/star_catalogs/*.npz list (same indexing
# as submit_batch.sh). Adjust --array=0-38 if the catalog count changes (39).

set -euo pipefail
cd /global/scratch/projects/pc_heptheory/kraman/DwarfJeansAnalysis

# Dynamic node exclusion: #SBATCH --exclude cannot interpolate, so when this
# script is launched directly (`bash ...`, no array task id yet) we re-exec
# sbatch with --exclude read from the shared bad-node list — the single source
# of truth. Extra CLI args (e.g. --array=0) are forwarded to sbatch; NO_PLOTS
# rides through on the exported environment. A bare `sbatch` launch sets
# SLURM_ARRAY_TASK_ID in the task, skips this branch, and falls back to the
# static #SBATCH --exclude above.
EXCLUDE_FILE=/global/scratch/projects/pc_heptheory/kraman/slurm_exclude_nodes.txt
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    excl_arg=()
    if [[ -f "$EXCLUDE_FILE" ]]; then
        excl="$(tr -d '[:space:]' < "$EXCLUDE_FILE")"
        if [[ -n "$excl" ]]; then excl_arg=(--exclude="$excl"); fi
    fi
    exec sbatch "${excl_arg[@]}" "$@" "$0"
fi

# conda init trips `set -eu` (unset $PS1, optional-file tests) — disable around it.
set +eu
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate DwarfJeans
set -eu

mapfile -t KEYS < <(ls data/star_catalogs/*.npz | xargs -n1 basename | sed 's/.npz$//' | sort)
KEY="${KEYS[$SLURM_ARRAY_TASK_ID]:-}"
if [[ -z "$KEY" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (${#KEYS[@]} catalogs)" >&2
    exit 1
fi

plots_flag=()
if [[ -n "${NO_PLOTS:-}" ]]; then
    plots_flag=(--no-plots)
fi

# HOST=mw2022|m12|stored selects the tidal-radius host for every run (rides
# through on the exported environment, like NO_PLOTS). Default 'stored' honors
# each chain's saved host; use HOST=mw2022 to roll the real-MW host onto the
# whole tree (legacy chains carry no host tag, so 'stored' == m12 for them).
host_flag=()
if [[ -n "${HOST:-}" ]]; then
    host_flag=(--host "$HOST")
fi

echo "Task $SLURM_ARRAY_TASK_ID: reprocessing all priors for lvdb_key=$KEY${NO_PLOTS:+ (no plots)}${HOST:+ (host=$HOST)}"
python scripts/reprocess.py --glob "$KEY/*" "${plots_flag[@]}" "${host_flag[@]}"
