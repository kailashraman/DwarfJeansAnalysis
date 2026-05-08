#!/bin/bash
#SBATCH --job-name=djprod
#SBATCH --time=4:00:00
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr7
#SBATCH --qos=lr_normal
#SBATCH --array=0-38
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --mail-type=NONE
#SBATCH --exclude=n0150.lr7
#SBATCH --output=/global/scratch/projects/pc_heptheory/kraman/DwarfJeansAnalysis/logs/slurm-%A_%a.out

# Production driver: one array task per staged catalog in
# data/star_catalogs/. Each task runs scripts/run_production.py for one
# lvdb_key with the Jeffreys prior. Outputs land under
# results/production/_slurm_<jobid>/_runs/<key>/jeffreys/<ts>/.
#
# Four ways to invoke:
#
#   sbatch scripts/submit_batch.sh
#       Submits the full --array=0-38 sweep over every staged catalog.
#
#   bash scripts/submit_batch.sh --cohort classical
#       The 10 heavy galaxies (N_post > 100) at pool=8 (header default).
#
#   bash scripts/submit_batch.sh --cohort ufd
#       The 29 light galaxies (N_post ≤ 100) at pool=1 (cheap likelihood;
#       pool overhead dominates so serial is more core-efficient).
#
#   bash scripts/submit_batch.sh [--pool N] draco_1 carina_1 leo_1 ...
#       Wrapper mode: looks up the array indices for the listed
#       lvdb_keys and submits an sbatch with --array overridden. Pool
#       size optional; defaults to the #SBATCH --cpus-per-task header.
#
# Adjust --array=0-38 if the catalog count changes (currently 39).
#
# Each task allocates --cpus-per-task=8 and the per-task body passes
# --npool=$SLURM_CPUS_PER_TASK to dynesty so the pool size auto-syncs
# with the SLURM allocation. Sized for the slowest classical (Draco-class):
# pool=8 expected to give ~4-4.5× wall reduction on those, ~3× on UFDs.
# Total array demand: 39 × 8 = 312 cores; lr7 should schedule in parallel.

set -euo pipefail
cd /global/scratch/projects/pc_heptheory/kraman/DwarfJeansAnalysis

# -----------------------------------------------------------------------------
# Wrapper mode: not running inside an array task, so translate user-supplied
# lvdb_keys into --array indices and resubmit.
# -----------------------------------------------------------------------------
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # Cohort lists. Membership is determined by post-cut N_post (i.e. N stars
    # surviving p_min, R/r_½, and variability cuts) and updated whenever the
    # underlying catalogs are re-staged. Threshold: N_post > 100 → CLASSICALS,
    # N_post ≤ 100 → UFDS. CLASSICALS get pool=8 (heavy likelihood), UFDS get
    # pool=1 (cheap likelihood, pool overhead dominates).
    CLASSICALS=(antlia_2 canes_venatici_1 carina_1 crater_2 draco_1
                leo_1 leo_2 sculptor_1 sextans_1 ursa_minor_1)
    UFDS=(aquarius_2 bootes_1 bootes_2 bootes_3
          canes_venatici_2 carina_2 carina_3 centaurus_1 coma_berenices_1
          eridanus_2 eridanus_4 grus_1 hercules_1 horologium_1 hydrus_1
          leo_4 leo_5 leo_6
          pegasus_3 pegasus_4 pisces_2 reticulum_2
          segue_1 tucana_2 tucana_4 tucana_5
          ursa_major_1 ursa_major_2 willman_1)

    # --cohort {classical|ufd} expands to the appropriate key list.
    # Auto-applies the recommended pool size unless --pool overrides.
    POOL_OVERRIDE=
    if [[ "${1:-}" == "--cohort" ]]; then
        case "${2:-}" in
            classical) set -- "${CLASSICALS[@]}" ;;
            ufd)       POOL_OVERRIDE=1; set -- "${UFDS[@]}" ;;
            *)
                echo "ERROR: --cohort must be 'classical' or 'ufd'" >&2
                exit 1 ;;
        esac
    fi

    # Optional --pool N override (defaults to #SBATCH --cpus-per-task above,
    # or to the cohort's recommended size if --cohort was used).
    if [[ "${1:-}" == "--pool" ]]; then
        POOL_OVERRIDE="$2"
        shift 2
    fi
    if [[ $# -eq 0 ]]; then
        echo "ERROR: invoke as one of:" >&2
        echo "       sbatch $0                          # full --array=0-38 sweep" >&2
        echo "       bash $0 --cohort classical         # 10 classicals at pool=8" >&2
        echo "       bash $0 --cohort ufd               # 29 UFDs at pool=1" >&2
        echo "       bash $0 [--pool N] KEY [KEY ...]   # explicit galaxy list" >&2
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
    sbatch_args=(--array="$array_arg")
    if [[ -n "$POOL_OVERRIDE" ]]; then
        sbatch_args+=(--cpus-per-task="$POOL_OVERRIDE")
        echo "Submitting subset: keys=$* -> --array=$array_arg --cpus-per-task=$POOL_OVERRIDE"
    else
        echo "Submitting subset: keys=$* -> --array=$array_arg"
    fi
    exec sbatch "${sbatch_args[@]}" "$0"
fi

# -----------------------------------------------------------------------------
# Per-task body — runs inside each SLURM array task.
# -----------------------------------------------------------------------------
# Sourcing ~/.bashrc and conda init both trip `set -eu`:
#   -u: /etc/bashrc references $PS1 (unset in batch shell)
#   -e: ~/.bashrc runs `test -f X` for optional group bashrcs that may not
#       exist; the test's exit-1 propagates and kills the script.
# Disable both around the conda init, then re-enable.
set +eu
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate DwarfJeans
set -eu

mapfile -t KEYS < <(ls data/star_catalogs/*.npz | xargs -n1 basename | sed 's/.npz$//' | sort)
KEY="${KEYS[$SLURM_ARRAY_TASK_ID]}"
NPOOL="${SLURM_CPUS_PER_TASK:-1}"
echo "Task $SLURM_ARRAY_TASK_ID: lvdb_key=$KEY  npool=$NPOOL"

OUT_BASE="results/production/_slurm_${SLURM_ARRAY_JOB_ID}/_runs"
mkdir -p "$OUT_BASE"

python scripts/run_production.py \
    --lvdb-key "$KEY" \
    --prior jeffreys \
    --nlive 500 \
    --dlogz 0.1 \
    --npool "$NPOOL" \
    --output-base "$OUT_BASE"
