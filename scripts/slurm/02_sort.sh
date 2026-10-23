#!/usr/bin/env bash
#SBATCH --job-name=ks4-sort
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=0-12:00
#SBATCH --output=logs/ks4-sort-%j.out
#SBATCH --error=logs/ks4-sort-%j.err
#
# Sort one recording with Kilosort 4: whole array, drift correction on, conditions concatenated.
#
#   sbatch 02_sort.sh <session-dir> <prefix>
#
# Output: $MEA_OUT/<prefix>. Needs ~65 GB GPU (A100 80GB) and ~476 GB scratch.
# --mem=250G: bombcell's analyzer needs > 128G.
# Env knobs:
#   MEA_N_JOBS=1         binary-write workers (default: allocated CPUs; output identical)
#   MEA_REUSE_SORT=1     relabel an existing sort instead of re-sorting
#   MEA_CONSTRAINT=...   GPU feature (default a100, set by submit_recording.sh; empty for none)
#   MEA_SORT_FLAGS=...   extra run_kilosort4.py options, e.g. --n-blocks 8 for a smaller card
set -euo pipefail

DATA_DIR="${1:?usage: sbatch 02_sort.sh <data-dir> <prefix>}"
STEM="${2:?usage: sbatch 02_sort.sh <data-dir> <prefix>}"

# Find the script dir; not dirname "$0": sbatch runs a spooled copy
# (no bash array: bash 4.2 on login nodes treats an empty one as unbound under set -u)
_self_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)"
HERE=""
for _c in "${MEA_SCRIPTS:-}" "${SLURM_SUBMIT_DIR:-}" "$_self_dir"; do
    if [[ -n "$_c" && -f "$_c/_common.sh" ]]; then HERE="$_c"; break; fi
done
[[ -n "$HERE" ]] || {
    echo "cannot find _common.sh. Tried MEA_SCRIPTS='${MEA_SCRIPTS:-}'," >&2
    echo "  SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}', '$_self_dir'." >&2
    echo "Set MEA_SCRIPTS to the directory holding these scripts." >&2
    exit 1; }
source "$HERE/_common.sh"
mea_paths
REPO="$(find_repo "$HERE")"


CUDA_MOD="$(module -t avail cuda 2>&1 | grep -E '^cuda/12\.' | sort -V | tail -1 || true)"
[[ -n "$CUDA_MOD" ]] && module load "$CUDA_MOD"
activate_env

export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-4}"   # bombcell spawns one worker per core
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

echo "=== $STEM, sort job ${SLURM_JOB_ID:-interactive} on $(hostname) ==="
# MEA_REUSE_SORT runs on CPU nodes without nvidia-smi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "no GPU on this node (expected with MEA_REUSE_SORT)"
fi
echo "data $DATA_DIR"
echo "out  $MEA_OUT/$STEM"

# Clear a previous attempt (Kilosort won't write into an existing folder), unless MEA_REUSE_SORT.
# Done before the free-space check so the old binary isn't counted.
RUN_DIR="$MEA_OUT/$STEM"
if [[ -n "${MEA_REUSE_SORT:-}" ]]; then
    compgen -G "$RUN_DIR/kilosort4_*" > /dev/null 2>&1 || {
        echo "ERROR: MEA_REUSE_SORT needs blocks already sorted, and" >&2
        echo "$RUN_DIR holds no kilosort4_* directory." >&2
        exit 1; }
    echo "reusing the sort in $RUN_DIR: relabelling, sorting nothing"
elif [[ -e "$RUN_DIR" ]]; then
    # Only ever the per-prefix dir, never $MEA_OUT
    [[ -n "$STEM" && "$RUN_DIR" == *"/$STEM" ]] || {
        echo "refusing to clear '$RUN_DIR': not a per-prefix directory" >&2; exit 1; }
    echo "clearing a previous attempt: $RUN_DIR"
    rm -rf "$RUN_DIR"
fi

# Refuse early if scratch can't hold the binary (shared filesystem)
mkdir -p "$MEA_OUT"
if [[ -n "${MEA_REUSE_SORT:-}" ]]; then
    echo "free-space check skipped: a reused sort writes no binary"
else
    NEED_GB=596
    AVAIL_GB=$(df -BG --output=avail "$MEA_OUT" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
    echo "free at $MEA_OUT: ${AVAIL_GB} GB, need about ${NEED_GB} GB"
    if [[ "$AVAIL_GB" -lt "$NEED_GB" ]]; then
        echo "ERROR: the sort needs ~${NEED_GB} GB and this filesystem has ${AVAIL_GB} GB." >&2
        echo "Point MEA_OUT at /n/netscratch, or free space." >&2
        exit 1
    fi
fi

srun python "$REPO/scripts/slurm/run_kilosort4.py" \
    --data "$DATA_DIR" \
    --stem "$STEM" \
    --out "$RUN_DIR" \
    ${MEA_SORT_FLAGS:-} \
    --n-jobs "${MEA_N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}" \
    ${MEA_REUSE_SORT:+--collect-only} \
    ${MEA_KEEP_DAT:+--keep-scratch}

echo
echo "=== what it actually used ==="
echo "  sacct -j ${SLURM_JOB_ID:-<jobid>} --format=JobID,Elapsed,MaxRSS,State"
