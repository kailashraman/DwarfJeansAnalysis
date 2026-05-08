#!/bin/bash
#SBATCH --job-name=djprod
#SBATCH --time=1:00:00
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
# Submit:  sbatch scripts/submit_batch.sh
# Keys are discovered at submit time (sorted lexicographically); index
# into them by SLURM_ARRAY_TASK_ID. Adjust --array=0-38 if the catalog
# count changes.

cd /global/scratch/projects/pc_heptheory/kraman/DwarfJeansAnalysis

source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate DwarfJeans

set -eo pipefail

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
