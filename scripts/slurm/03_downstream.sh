#!/usr/bin/env bash
#SBATCH --job-name=downstream
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-04:00
#SBATCH --output=logs/ks4-downstream-%j.out
#SBATCH --error=logs/ks4-downstream-%j.err
#
# Run everything after the sort for one recording (mea plan, mea run), then archive.
#
#   sbatch --dependency=afterok:<sort-job> 03_downstream.sh <session-dir> <prefix>
#
# Archives even if `mea run` fails, then exits with its status. A job killed for
# time or memory never archives: run 05_archive.sh by hand.
set -euo pipefail

DATA_DIR="${1:?usage: sbatch 03_downstream.sh <data-dir> <prefix>}"
STEM="${2:?usage: sbatch 03_downstream.sh <data-dir> <prefix>}"

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

SORTED="$MEA_OUT/$STEM"
RESULTS="$MEA_RESULTS/$STEM"

activate_env
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

echo "=== $STEM, downstream job ${SLURM_JOB_ID:-interactive} on $(hostname) ==="

rc=0
echo; echo "=== 1. the sorter output is shaped as expected ==="
mea plan --input "$SORTED" || rc=$?
if [[ $rc -eq 0 ]]; then
    echo; echo "=== 2. QC, metrics, figures ==="
    mea run --input "$SORTED" --out "$RESULTS" || rc=$?
fi

cat <<MSG

  sorted output   $SORTED
  results         $RESULTS
  figures         $RESULTS/*.mea/figures
  compiled tables $RESULTS/compiled/

  resources:  sacct -j ${SLURM_JOB_ID:-<jobid>} --format=JobID,Elapsed,MaxRSS,State

MSG
[[ $rc -eq 0 ]] || echo "mea run FAILED (exit $rc); archiving the sort anyway"

echo; echo "=== 3. archive ==="
if ! bash "$HERE/05_archive.sh" "$STEM"; then
    echo "archive FAILED" >&2
    [[ $rc -ne 0 ]] || rc=1
fi
exit "$rc"
