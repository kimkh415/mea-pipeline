#!/usr/bin/env bash
#
# Submit a list of recordings.  Run from a LOGIN node; submits, does not compute.
#
#   bash submit_samples.sh SAMPLES.tsv --dry-run   # check every line, submit nothing
#   bash submit_samples.sh SAMPLES.tsv             # one submit_recording.sh per line
#
# SAMPLES.tsv: one recording per line, tab-separated, header optional:
#
#   directory<TAB>prefix
#   /path/to/recording_directory<TAB>392_het_100Hz
#
# Lines starting with # are skipped. Directories are given, not searched (duplicates exist).
#
# MEA_SCRATCH and MEA_FINAL are required, first from --scratch/--final, then mea.env
# beside SAMPLES.tsv, then the environment. mea.env may set any MEA_* variable.
# logs/ and a copy of this output (logs/submit_<time>.txt) go beside SAMPLES.tsv.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage="usage: bash submit_samples.sh SAMPLES.tsv [--dry-run] [--scratch DIR] [--final DIR]"

LIST=""; DRY=0; ARG_SCRATCH=""; ARG_FINAL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n) DRY=1 ;;
        --scratch) ARG_SCRATCH="${2:?--scratch needs a directory}"; shift ;;
        --final)   ARG_FINAL="${2:?--final needs a directory}"; shift ;;
        -*) echo "unknown option: $1" >&2; echo "$usage" >&2; exit 2 ;;
        *) [[ -z "$LIST" ]] || { echo "one sample list only: '$LIST' and '$1'" >&2; exit 2; }
           LIST="$1" ;;
    esac
    shift
done
[[ -n "$LIST" ]] || { echo "$usage" >&2; exit 2; }
[[ -f "$LIST" ]] || { echo "no sample list at $LIST" >&2; echo "$usage" >&2; exit 2; }
LIST="$(cd "$(dirname "$LIST")" && pwd)/$(basename "$LIST")"
PROJECT="$(dirname "$LIST")"

SRC_SCRATCH="environment"; SRC_FINAL="environment"
unset MEA_OUT MEA_RESULTS                     # re-derived from MEA_SCRATCH unless mea.env sets them
if [[ -f "$PROJECT/mea.env" ]]; then
    # shellcheck disable=SC1091
    set -a; source "$PROJECT/mea.env"; set +a  # export, so the jobs see them
    sets() { grep -qE "^[[:space:]]*(export[[:space:]]+)?$1=" "$PROJECT/mea.env"; }
    sets MEA_SCRATCH && SRC_SCRATCH="mea.env"
    sets MEA_FINAL && SRC_FINAL="mea.env"
fi
[[ -n "$ARG_SCRATCH" ]] && { MEA_SCRATCH="$ARG_SCRATCH"; SRC_SCRATCH="--scratch"; }
[[ -n "$ARG_FINAL" ]]   && { MEA_FINAL="$ARG_FINAL";     SRC_FINAL="--final"; }
missing=""
[[ -n "${MEA_SCRATCH:-}" ]] || missing="$missing MEA_SCRATCH"
[[ -n "${MEA_FINAL:-}" ]]   || missing="$missing MEA_FINAL"
if [[ -n "$missing" ]]; then
    echo "not set:$missing" >&2
    echo "Put them in $PROJECT/mea.env:" >&2
    echo "    MEA_SCRATCH=/n/netscratch/arlotta_lab/Users/\$USER/mea   # work directory" >&2
    echo "    MEA_FINAL=/path/on/lab/storage/results                  # kept results" >&2
    exit 2
fi
export MEA_SCRATCH MEA_FINAL

source "$HERE/_common.sh"
mea_paths

# Check write access before queueing
for d in "$MEA_SCRATCH" "$MEA_FINAL" "$PROJECT/logs"; do
    { mkdir -p "$d" && touch "$d/.write_test" && rm -f "$d/.write_test"; } 2>/dev/null || {
        echo "cannot write to $d" >&2; exit 1; }
done

cd "$PROJECT"                                 # sbatch writes logs/ relative to here
exec > >(tee "logs/submit_$(date +%Y%m%d_%H%M%S)$([[ $DRY -eq 1 ]] && echo _dryrun).txt") 2>&1

echo "list      $LIST"
echo "scratch   $MEA_SCRATCH   ($SRC_SCRATCH)"
echo "final     $MEA_FINAL   ($SRC_FINAL)"
echo "logs      $PROJECT/logs"
[[ $DRY -eq 1 ]] && echo "MODE      dry run -- nothing will be submitted"
echo

ok=0; bad=0; n=0; seen=$'\n'
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"                       # a list saved on Windows
    [[ -z "${line//[[:space:]]/}" || "$line" == \#* || "$line" == directory$'\t'* || "$line" == session$'\t'* ]] && continue
    n=$((n+1))
    if [[ "$line" != *$'\t'* ]]; then
        echo "=== line $n: no tab"
        echo "  ERROR: expected directory<TAB>prefix, got: $line"
        bad=$((bad+1)); echo; continue
    fi
    IFS=$'\t' read -r directory prefix _ <<< "$line"
    echo "=== $prefix"
    echo "  directory  $directory"
    if [[ -z "$prefix" ]]; then
        echo "  ERROR: no prefix after the tab"; bad=$((bad+1)); echo; continue
    fi
    # A duplicate prefix would clear the first one's running sort
    if [[ "$seen" == *$'\n'"$prefix"$'\n'* ]]; then
        echo "  ERROR: $prefix is listed twice; it would overwrite itself"; bad=$((bad+1)); echo; continue
    fi
    seen="$seen$prefix"$'\n'
    if [[ ! -d "$directory" ]]; then
        echo "  ERROR: no such directory"; bad=$((bad+1)); echo; continue
    fi
    nbrw=$(find "$directory" -maxdepth 1 -name "$prefix*.brw" | wc -l | tr -d ' ')
    find "$directory" -maxdepth 1 -name "$prefix*.brw" | sort | while IFS= read -r f; do
        printf "           %6s  %s\n" "$(ls -lh "$f" | awk '{print $5}')" "$(basename "$f")"
    done
    if [[ "$nbrw" -lt 2 ]]; then
        echo "  ERROR: $nbrw .brw file(s) match $prefix*; need a baseline plus TTX at least"
        bad=$((bad+1)); echo; continue
    fi
    if [[ $DRY -eq 0 ]]; then
        MEA_SESSION="$directory" MEA_STEM="$prefix" bash "$HERE/submit_recording.sh"
    fi
    ok=$((ok+1)); echo
done < "$LIST"

if [[ $n -eq 0 ]]; then
    echo "no recordings in $LIST (only blank, comment or header lines)" >&2
    exit 1
fi
echo "$ok recording(s) ok, $bad with errors"
if [[ $DRY -eq 1 ]]; then
    echo "re-run without --dry-run to submit"
fi
[[ $bad -eq 0 ]]
