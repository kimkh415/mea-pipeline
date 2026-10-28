#!/usr/bin/env bash
#
# Submit one recording: sort job (GPU), then downstream job (CPU) that archives. Run on a login node.
# Normally called by submit_samples.sh; directly:
#
#   MEA_SESSION=<session-dir> MEA_STEM=<prefix> bash submit_recording.sh
set -euo pipefail

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
# Resolve the repo now; exported for the jobs
export MEA_REPO="${MEA_REPO:-$(find_repo "$HERE")}"
# Jobs can't locate these scripts themselves (spooled copy)
export MEA_SCRIPTS="$HERE"

SESSION="${MEA_SESSION:?set MEA_SESSION to the folder holding the .brw files}"
STEM="${MEA_STEM:?set MEA_STEM to the recording prefix}"
# GPU feature; empty for none (an unknown feature pends forever)
MEA_CONSTRAINT="${MEA_CONSTRAINT-a100}"
# MEA_MAIL: sort mails on failure, downstream on end
MAIL_FAIL="${MEA_MAIL:+--mail-user=$MEA_MAIL --mail-type=FAIL,TIME_LIMIT}"
MAIL_END="${MEA_MAIL:+--mail-user=$MEA_MAIL --mail-type=END,FAIL}"

mea_paths

mkdir -p logs "$MEA_OUT" "$MEA_RESULTS"

echo "directory $SESSION"
echo "prefix    $STEM"
echo "sorted    $MEA_OUT/$STEM"
echo "results   $MEA_RESULTS/$STEM"
echo "final     $MEA_FINAL/$STEM"
[[ -n "${MEA_REUSE_SORT:-}" ]] && echo "mode      reuse sort: relabel the existing sort, no GPU"
echo "env       $(env_target)"

# Check inputs before queueing
[[ -d "$SESSION" ]] || { echo "no such directory: $SESSION" >&2; exit 1; }
N=$(find "$SESSION" -maxdepth 1 -name "$STEM*.brw" | wc -l | tr -d ' ')
MIN_BRW="${MEA_MIN_BRW:-2}"     # 1 for single-condition runs
[[ "$N" -ge "$MIN_BRW" ]] || { echo "expected at least $MIN_BRW .brw files for $STEM, found $N (MEA_MIN_BRW)" >&2; exit 1; }
env_exists || {
    echo "no conda environment '$(env_target)' -- run 01_setup_env.sh first." >&2
    echo "It installs conda if the cluster has none, then builds the env." >&2
    exit 1; }

# A reused sort needs no GPU; use a large-memory CPU partition (shared's 188 GB nodes are too small).
# MEA_SORT_MEM overrides the 250G default.
if [[ -n "${MEA_REUSE_SORT:-}" ]]; then
    SORT_RESOURCES="--partition=${MEA_CPU_PARTITION:-sapphire} --gpus=0"
    SORT_RESOURCES="$SORT_RESOURCES --cpus-per-task=${MEA_CPUS:-16}"
    SORT_RESOURCES="$SORT_RESOURCES --mem=${MEA_SORT_MEM:-250G}"
else
    SORT_RESOURCES="${MEA_CONSTRAINT:+--constraint=$MEA_CONSTRAINT}"
    SORT_RESOURCES="$SORT_RESOURCES ${MEA_SORT_MEM:+--mem=$MEA_SORT_MEM}"
fi

# Job names carry the prefix; a failed sort cancels the downstream job
SID=$(sbatch --parsable --job-name="sort_$STEM" $SORT_RESOURCES $MAIL_FAIL \
             "$HERE/02_sort.sh" "$SESSION" "$STEM")
RID=$(sbatch --parsable --job-name="downstream_$STEM" --dependency="afterok:$SID" \
             --kill-on-invalid-dep=yes $MAIL_END \
             "$HERE/03_downstream.sh" "$SESSION" "$STEM")
echo "jobs      sort $SID, downstream $RID"
