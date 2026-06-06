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
#SBATCH --output=/global/scratch/projects/pc_heptheory/kraman/DwarfJeansAnalysis/logs/reproc-%A_%a.out

# Regenerate the derived views (summary.csv + plots) for every existing
# production run WITHOUT re-running dynesty. One array task per galaxy; each
# task reprocesses all of that galaxy's priors via scripts/reprocess.py
# --glob "<key>/*". Derived quantities are recomputed from the saved chain
# (and backfilled from audit.json for legacy npz), so this is the cheap way to
# roll the new tidal-radius / theta95 quantities onto the whole results tree.
#
# Cost: ~8 priors x ~50 s/run ~= 7 min per task; 39 tasks run in parallel.
#
# Usage:
#   sbatch scripts/submit_reprocess.sh                 # full 39-galaxy sweep
#   sbatch --array=0 scripts/submit_reprocess.sh       # one galaxy (index 0)
#   NO_PLOTS=1 sbatch scripts/submit_reprocess.sh      # summary.csv only (faster)
#
# Array indices match the sorted data/star_catalogs/*.npz list (same indexing
# as submit_batch.sh). Adjust --array=0-38 if the catalog count changes (39).

set -euo pipefail
cd /global/scratch/projects/pc_heptheory/kraman/DwarfJeansAnalysis

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

echo "Task $SLURM_ARRAY_TASK_ID: reprocessing all priors for lvdb_key=$KEY${NO_PLOTS:+ (no plots)}"
python scripts/reprocess.py --glob "$KEY/*" "${plots_flag[@]}"
