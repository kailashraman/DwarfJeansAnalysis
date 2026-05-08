#!/bin/bash
#SBATCH --job-name=djprod
#SBATCH --time=4:00:00
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr7
#SBATCH --qos=lr_normal
#SBATCH --array=0-38
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=NONE
#SBATCH --output=/global/scratch/projects/pc_heptheory/kraman/DwarfJeansAnalysis/logs/slurm-%A_%a.out

# Production driver: one array task per staged catalog in
# data/star_catalogs/. Each task runs scripts/run_production.py for one
# lvdb_key with the Jeffreys prior. Outputs land under
# results/production/_slurm_<jobid>/_runs/<key>/jeffreys/<ts>/.
#
# Two ways to invoke:
#
#   sbatch scripts/submit_batch.sh
#       Submits the full --array=0-38 sweep over every staged catalog.
#
#   bash scripts/submit_batch.sh draco_1 carina_1 leo_1 ursa_minor_1
#       Wrapper mode: looks up the array indices for the listed
#       lvdb_keys and submits an sbatch with --array overridden to
#       just those indices. Same #SBATCH resources (time, partition,
#       memory) as the full sweep.
#
# Adjust --array=0-38 if the catalog count changes (currently 39).

set -euo pipefail
cd /global/scratch/projects/pc_heptheory/kraman/DwarfJeansAnalysis

# -----------------------------------------------------------------------------
# Wrapper mode: not running inside an array task, so translate user-supplied
# lvdb_keys into --array indices and resubmit.
# -----------------------------------------------------------------------------
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if [[ $# -eq 0 ]]; then
        echo "ERROR: invoke either as 'sbatch $0' for the full sweep, or as" >&2
        echo "       'bash $0 KEY [KEY ...]' to submit only those galaxies." >&2
        exit 1
    fi
    mapfile -t ALL_KEYS < <(ls data/star_catalogs/*.npz | xargs -n1 basename | sed 's/.npz$//' | sort)
    INDICES=()
    for want in "$@"; do
        found=
        for i in "${!ALL_KEYS[@]}"; do
            if [[ "${ALL_KEYS[$i]}" == "$want" ]]; then
                INDICES+=("$i"); found=1; break
            fi
        done
        if [[ -z "$found" ]]; then
            echo "ERROR: lvdb_key '$want' not in data/star_catalogs/" >&2
            exit 2
        fi
    done
    array_arg=$(IFS=,; echo "${INDICES[*]}")
    echo "Submitting subset: keys=$* -> --array=$array_arg"
    exec sbatch --array="$array_arg" "$0"
fi

# -----------------------------------------------------------------------------
# Per-task body — runs inside each SLURM array task.
# -----------------------------------------------------------------------------
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate DwarfJeans

mapfile -t KEYS < <(ls data/star_catalogs/*.npz | xargs -n1 basename | sed 's/.npz$//' | sort)
KEY="${KEYS[$SLURM_ARRAY_TASK_ID]}"
echo "Task $SLURM_ARRAY_TASK_ID: lvdb_key=$KEY"

OUT_BASE="results/production/_slurm_${SLURM_ARRAY_JOB_ID}/_runs"
mkdir -p "$OUT_BASE"

python scripts/run_production.py \
    --lvdb-key "$KEY" \
    --prior jeffreys \
    --nlive 500 \
    --dlogz 0.1 \
    --output-base "$OUT_BASE"
