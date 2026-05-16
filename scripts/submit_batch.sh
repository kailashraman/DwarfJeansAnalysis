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
# lvdb_key. Outputs land under the canonical
# results/production/<key>/<prior>/ and overwrite the previous run.
#
# Sampler / prior config (env overrides, exported through sbatch):
#   PRIOR  (default: jeffreys)   — one of {jeffreys, loguniform, uniform,
#                                  satgen, satgen_box, satgen_shmr}
#   SHMR   (default: <unset>)    — required iff PRIOR=satgen_shmr; one of
#                                  the choices in run_production.py --shmr
#   NLIVE  (default: 1500)       — dynesty nlive (dense production)
#   DLOGZ  (default: 0.05)       — dynesty dlogz stop (dense production)
# To reproduce the prior 500/0.1 sampling: NLIVE=500 DLOGZ=0.1 sbatch ...
# To run both priors: submit twice with PRIOR=jeffreys and PRIOR=loguniform.
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

    # Sampler/prior CLI flags. These get baked into the sbatch --export so
    # the array tasks see them regardless of the submitting shell's env.
    # Flag order is irrelevant: keys (including those expanded from
    # --cohort) accumulate into KEYS rather than mutating $@ mid-parse, so
    # `--cohort classical --prior jeffreys` and `--prior jeffreys --cohort
    # classical` are equivalent.
    PRIOR_ARG=jeffreys
    SHMR_ARG=
    NLIVE_ARG=1500
    DLOGZ_ARG=0.05
    POOL_OVERRIDE=
    COHORT=
    KEYS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --prior)  PRIOR_ARG="$2"; shift 2 ;;
            --shmr)   SHMR_ARG="$2"; shift 2 ;;
            --nlive)  NLIVE_ARG="$2"; shift 2 ;;
            --dlogz)  DLOGZ_ARG="$2"; shift 2 ;;
            --pool)   POOL_OVERRIDE="$2"; shift 2 ;;
            --cohort) COHORT="$2"; shift 2 ;;
            --) shift; while [[ $# -gt 0 ]]; do KEYS+=("$1"); shift; done ;;
            -h|--help)
                cat >&2 <<EOF
Usage:
  bash $0 [flags]                     # full 39-galaxy sweep
  bash $0 [flags] --cohort classical  # 10 classicals at pool=8
  bash $0 [flags] --cohort ufd        # 29 UFDs at pool=1
  bash $0 [flags] KEY [KEY ...]       # explicit galaxy list

Flags (order-independent):
  --prior {jeffreys|loguniform|uniform|satgen|satgen_box|satgen_shmr}
                                          default: jeffreys
  --shmr  {fattahi18|...}                 required iff --prior satgen_shmr
  --nlive INT                             default: 1500
  --dlogz FLOAT                           default: 0.05
  --pool INT                              override --cpus-per-task
  --cohort {classical|ufd}                expand into the cohort key list
EOF
                exit 0 ;;
            -*) echo "ERROR: unknown flag $1" >&2; exit 1 ;;
            *)  KEYS+=("$1"); shift ;;
        esac
    done

    # Expand --cohort into the KEYS list.
    if [[ -n "$COHORT" ]]; then
        case "$COHORT" in
            classical) KEYS=("${CLASSICALS[@]}" "${KEYS[@]}") ;;
            ufd)
                POOL_OVERRIDE="${POOL_OVERRIDE:-1}"
                KEYS=("${UFDS[@]}" "${KEYS[@]}")
                ;;
            *) echo "ERROR: --cohort must be 'classical' or 'ufd' (got '$COHORT')" >&2
               exit 1 ;;
        esac
    fi

    mapfile -t ALL_KEYS < <(ls data/star_catalogs/*.npz | xargs -n1 basename | sed 's/.npz$//' | sort)
    if [[ ${#KEYS[@]} -eq 0 ]]; then
        # Full sweep — all staged catalogs.
        INDICES=()
        for i in "${!ALL_KEYS[@]}"; do INDICES+=("$i"); done
    else
        INDICES=()
        for want in "${KEYS[@]}"; do
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
    fi
    if [[ "$PRIOR_ARG" == "satgen_shmr" && -z "$SHMR_ARG" ]]; then
        echo "ERROR: --prior satgen_shmr requires --shmr" >&2; exit 1
    fi
    if [[ -n "$SHMR_ARG" && "$PRIOR_ARG" != "satgen_shmr" ]]; then
        echo "ERROR: --shmr is only valid with --prior satgen_shmr" >&2; exit 1
    fi
    array_arg=$(IFS=,; echo "${INDICES[*]}")
    export_list="ALL,PRIOR=$PRIOR_ARG,NLIVE=$NLIVE_ARG,DLOGZ=$DLOGZ_ARG"
    if [[ -n "$SHMR_ARG" ]]; then
        export_list+=",SHMR=$SHMR_ARG"
    fi
    sbatch_args=(--array="$array_arg" --export="$export_list")
    if [[ -n "$POOL_OVERRIDE" ]]; then
        sbatch_args+=(--cpus-per-task="$POOL_OVERRIDE")
    fi
    echo "Submitting: prior=$PRIOR_ARG${SHMR_ARG:+ shmr=$SHMR_ARG}" \
         "nlive=$NLIVE_ARG dlogz=$DLOGZ_ARG" \
         "pool=${POOL_OVERRIDE:-<header default>} array=$array_arg"
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

# Sampler config is overridable via env so the same script drives multiple
# prior/sampling sweeps without further edits. Defaults are the dense
# production setting; the original (500, 0.1) is recoverable by exporting
# NLIVE/DLOGZ at submit time.
PRIOR="${PRIOR:-jeffreys}"
SHMR="${SHMR:-}"
NLIVE="${NLIVE:-1500}"
DLOGZ="${DLOGZ:-0.05}"
echo "Task $SLURM_ARRAY_TASK_ID: lvdb_key=$KEY  npool=$NPOOL  prior=$PRIOR${SHMR:+  shmr=$SHMR}  nlive=$NLIVE  dlogz=$DLOGZ"

shmr_flag=()
if [[ -n "$SHMR" ]]; then
    shmr_flag=(--shmr "$SHMR")
fi

python scripts/run_production.py \
    --lvdb-key "$KEY" \
    --prior "$PRIOR" \
    "${shmr_flag[@]}" \
    --nlive "$NLIVE" \
    --dlogz "$DLOGZ" \
    --npool "$NPOOL"
