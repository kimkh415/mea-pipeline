#!/usr/bin/env bash
#SBATCH --job-name=ks4-archive
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-06:00
#SBATCH --output=logs/ks4-archive-%j.out
#SBATCH --error=logs/ks4-archive-%j.err
#
# Copy one recording from scratch to $MEA_FINAL/<prefix>/ (called by 03_downstream.sh).
# Run by hand from the project folder if a downstream job was killed:
#
#   sbatch 05_archive.sh <prefix>
#
# Layout:
#   figures/                 diagnostic PDFs
#   compiled_metrics/        compiled.wide.csv for this recording
#   pipeline/
#     sort/<prefix>/         sorter output + kilosort4_<n>.tar.gz (no recording.dat)
#     <prefix>.mea/          the .mea store
#     logs/                  this recording's job logs
#     ARCHIVE.txt
#
# Copies only; MEA_ARCHIVE_PURGE=1 deletes the scratch sort afterwards.
set -euo pipefail

STEM="${1:?usage: sbatch 05_archive.sh <prefix>}"

# Find the script dir; not dirname "$0": sbatch runs a spooled copy
_self_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)"
HERE=""
for _c in "${MEA_SCRIPTS:-}" "${SLURM_SUBMIT_DIR:-}" "$_self_dir"; do
    if [[ -n "$_c" && -f "$_c/_common.sh" ]]; then HERE="$_c"; break; fi
done
[[ -n "$HERE" ]] || {
    echo "cannot find _common.sh. Tried MEA_SCRIPTS='${MEA_SCRIPTS:-}'," >&2
    echo "  SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}', '$_self_dir'." >&2
    exit 1; }
source "$HERE/_common.sh"
mea_paths

SRC="$MEA_OUT/$STEM"
RES="$MEA_RESULTS/$STEM"
DEST="$MEA_FINAL/$STEM"
LOGS="${SLURM_SUBMIT_DIR:-$PWD}/logs"   # the project folder's, where sbatch wrote them
echo "=== archiving $STEM ==="
echo "from  $SRC"
echo "to    $DEST"

# Require sort_provenance.json, which run_kilosort4.py writes last
[[ -d "$SRC" ]] || { echo "ERROR: no sort for $STEM at $SRC; nothing copied" >&2; exit 1; }
[[ -f "$SRC/sort_provenance.json" ]] || {
    echo "ERROR: $SRC has no sort_provenance.json, which run_kilosort4.py" >&2
    echo "writes last: the sort is unfinished or failed.  Nothing copied." >&2
    exit 1; }

# Upper bound on size, excluding .dat (|| true: du fails if results dir is missing)
WANT_KB=$(du -sk --exclude='*.dat' "$SRC" "$RES" 2>/dev/null | awk '{s+=$1} END {print s+0}' || true)
WANT_GB=$(( ${WANT_KB:-0} / 1048576 + 1 ))
mkdir -p "$DEST"
AVAIL_GB=$(df -BG --output=avail "$DEST" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
echo "need at most ${WANT_GB} GB, ${AVAIL_GB} GB free"
if [[ "${AVAIL_GB:-0}" -lt "$((WANT_GB + 5))" ]]; then
    echo "ERROR: not enough room at $DEST." >&2
    echo "Lab storage is shared; free space or set MEA_FINAL elsewhere." >&2
    exit 1
fi

# Prefer pigz (parallel gzip)
if command -v pigz >/dev/null 2>&1; then
    ZIP=(-I "pigz -p ${SLURM_CPUS_PER_TASK:-4}")
else
    ZIP=(-z)
fi

P="$DEST/pipeline"
S="$P/sort/$STEM"
mkdir -p "$S" "$DEST/figures" "$DEST/compiled_metrics"

# Copy flat sorter outputs and check sizes
n=0
for f in "$SRC"/*.csv "$SRC"/*.npy "$SRC"/*.json; do
    [[ -f "$f" ]] || continue
    cp -p "$f" "$S/"
    [[ "$(stat -c%s "$f")" == "$(stat -c%s "$S/$(basename "$f")")" ]] || {
        echo "ERROR: $(basename "$f") copied short" >&2; exit 1; }
    n=$((n + 1))
done
echo "  pipeline/sort/$STEM/ $n files"

# Tar each sorter folder without the binary (--exclude must precede the path)
for d in "$SRC"/kilosort4_*; do
    [[ -d "$d" ]] || continue
    b="$(basename "$d")"
    tar --exclude='*.dat' "${ZIP[@]}" -cf "$S/$b.tar.gz" -C "$SRC" "$b"
    # Verify by reading back
    tar -tf "$S/$b.tar.gz" >/dev/null || {
        echo "ERROR: $b.tar.gz does not read back" >&2; exit 1; }
    echo "  pipeline/sort/$STEM/$b.tar.gz $(du -sh "$S/$b.tar.gz" | cut -f1)"
done

# Copy the .mea store; lift figures and compiled table to the top (/. avoids nesting on re-run)
if [[ -d "$RES" ]]; then
    for m in "$RES"/*.mea; do
        [[ -d "$m" ]] || continue
        cp -rp "$m" "$P/"
        if [[ -d "$m/figures" ]]; then
            cp -rp "$m/figures/." "$DEST/figures/"
            rm -rf "$P/$(basename "$m")/figures"
        fi
    done
    [[ -d "$RES/compiled" ]] && cp -rp "$RES/compiled/." "$DEST/compiled_metrics/"
    echo "  pipeline/*.mea $(du -sh "$P"/*.mea 2>/dev/null | cut -f1 | head -1)," \
         "figures/ $(ls "$DEST/figures" | wc -l | tr -d ' ') files," \
         "compiled_metrics/ $(ls "$DEST/compiled_metrics" | wc -l | tr -d ' ') file(s)"
else
    echo "  NOTE: no downstream output at $RES"
fi

# Copy logs whose first line names this prefix (running job's log is a snapshot)
if [[ -d "$LOGS" ]]; then
    mkdir -p "$P/logs"
    nl=0
    for f in "$LOGS"/ks4-*.out; do
        [[ -f "$f" ]] || continue
        case "$(head -1 "$f")" in
            "=== $STEM, "*|"=== archiving $STEM ==="*)
                cp -p "$f" "${f%.out}.err" "$P/logs/" 2>/dev/null || cp -p "$f" "$P/logs/"
                nl=$((nl + 1)) ;;
        esac
    done
    echo "  pipeline/logs/ $nl job(s)"
fi

{
    echo "prefix       $STEM"
    echo "archived     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "from         $SRC"
    echo "job          ${SLURM_JOB_ID:-interactive}"
    echo "excluded     recording.dat (rebuildable from the .brw files)"
} > "$P/ARCHIVE.txt"

if [[ -n "${MEA_ARCHIVE_PURGE:-}" ]]; then
    [[ -n "$STEM" && "$SRC" == *"/$STEM" ]] || {
        echo "refusing to purge '$SRC': not a per-prefix directory" >&2; exit 1; }
    echo "  MEA_ARCHIVE_PURGE set, removing $SRC"
    rm -rf "$SRC"
fi

echo "=== archived to $DEST: $(du -sh "$DEST" | cut -f1) ==="
