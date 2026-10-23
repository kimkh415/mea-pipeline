#!/usr/bin/env bash
#SBATCH --job-name=ks4-setup
#SBATCH --partition=gpu_test
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00
#SBATCH --output=logs/ks4-setup-%j.out
#SBATCH --error=logs/ks4-setup-%j.err
#
# Build the Kilosort 4 environment; installs conda and clones the repo if absent.
# Run once, interactively on a GPU node:
#
#   salloc -p gpu_test --gpus 1 -c 8 --mem 32G -t 0-04:00
#   bash 01_setup_env.sh
#
# Needs a GPU: pip picks the CUDA torch build from the visible driver.
set -euo pipefail

PY_VERSION=3.11
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

echo "=== node and driver ==="; hostname
if ! nvidia-smi; then
    echo >&2
    echo "No GPU visible here." >&2
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "This looks like a login node.  pip chooses between the CUDA and" >&2
        echo "CPU builds of torch by looking at the driver it can see, so" >&2
        echo "installing here gets you a CPU build that fails at minute forty" >&2
        echo "of a sort rather than now.  Get a session first:" >&2
        echo >&2
        echo "    salloc -p gpu_test --gpus 1 -c 8 --mem 32G -t 0-04:00" >&2
        echo "    bash $0" >&2
        echo >&2
        echo "(older form: srun --pty -p gpu_test --gpus 1 -c 8 --mem 32G \\" >&2
        echo "                  -t 0-04:00 /bin/bash)" >&2
    else
        echo "You are in a job, but it has no GPU.  Ask for one with --gpus 1" >&2
        echo "on a gpu partition." >&2
    fi
    echo >&2
    echo "To build a CPU-only environment anyway, set MEA_ALLOW_NO_GPU=1." >&2
    [[ "${MEA_ALLOW_NO_GPU:-0}" == "1" ]] || exit 1
    echo "MEA_ALLOW_NO_GPU=1: continuing without a GPU." >&2
fi

echo; echo "=== the repository ==="
REPO="$(ensure_repo "$HERE")"
echo "repo $REPO"

echo; echo "=== conda ==="
ensure_conda                                  # finds it, or installs Miniforge

CUDA_MOD="$(module -t avail cuda 2>&1 | grep -E '^cuda/12\.' | sort -V | tail -1 || true)"
[[ -n "$CUDA_MOD" ]] && { echo "loading $CUDA_MOD"; module load "$CUDA_MOD"; }

echo; echo "=== environment ==="
ENV_TARGET="$(env_target)"
if env_exists; then
    echo "environment '$ENV_TARGET' exists -- remove it to rebuild:"
    echo "    conda env remove $(env_flag) $ENV_TARGET"
else
    # Include pip, or installs below go to base's pip
    conda create -y "$(env_flag)" "$ENV_TARGET" "python=$PY_VERSION" pip
fi
conda activate "$ENV_TARGET"
# Record the env for later jobs
echo "$ENV_TARGET" > "$HERE/.mea_env" 2>/dev/null || true

# Refuse if activation left base python active
ACTIVE="$(python -c 'import sys; print(sys.prefix)')"
echo "active python: $ACTIVE"
[[ "$ACTIVE" == *"$ENV_TARGET"* ]] || {
    echo "activated '$ENV_TARGET' but python is $ACTIVE -- refusing to install" >&2
    echo "into the wrong environment." >&2; exit 1; }

python -m pip install --upgrade pip

# torch first and alone, so a dependency can't pull in a CPU build
python -m pip install "torch==2.14.0+cu126" --index-url https://download.pytorch.org/whl/cu126

# Validated pins. neo >= 0.14.3 shifts the noise baseline (gap fill works around it);
# numpy shares an ABI with torch/kilosort/bombcell.
python -m pip install \
    "kilosort==4.1.7" \
    "spikeinterface[full]==0.104.9" \
    "neo==0.14.5" \
    "bombcell==0.89" \
    "numpy==2.4.6" \
    "pandas>=2.2" "scipy>=1.13" "h5py>=3.11" "pyarrow>=16" "matplotlib>=3.8"

# Pipeline, editable: the sorter imports mea.io.brw_fill, and spikeinterface needs its installed version
python -m pip install -e "$REPO"

echo; echo "=== verify ==="
python - <<'PY'
import os, sys, torch, kilosort, spikeinterface, neo, numpy, mea
from packaging.version import parse
print(f"python         {sys.version.split()[0]}")
for name, mod in (("torch", torch), ("kilosort", kilosort),
                  ("spikeinterface", spikeinterface), ("neo", neo),
                  ("numpy", numpy), ("mea-pipeline", mea)):
    print(f"{name:14s} {mod.__version__}")
import bombcell
print(f"{'bombcell':14s} {getattr(bombcell, '__version__', '(installed)')}")
from spikeinterface.sorters import run_sorter                      # noqa: F401
from spikeinterface.extractors import read_kilosort                # noqa: F401
print("spikeinterface: run_sorter and read_kilosort import")

# Fail on a CPU torch build
if os.environ.get("MEA_ALLOW_NO_GPU") == "1" and not torch.cuda.is_available():
    print("MEA_ALLOW_NO_GPU=1: torch has no GPU. This environment CANNOT sort.")
else:
    assert torch.cuda.is_available(), "torch cannot see the GPU: this is a CPU build"
if torch.cuda.is_available():
    print(f"{'gpu':14s} {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"{'gpu memory':14s} {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
# GPU size is not checked here (gpu_test gives MIG slices); 02_sort.sh checks it

# spikeinterface needs a PEP 440 mea version to reload the recording
parse(mea.__version__)
print(f"{'mea version':14s} {mea.__version__} (parses)")

from spikeinterface.extractors import read_biocam       # fail here, not at minute forty
from mea.io.brw_fill import open_brw_with_noise_fill
print("read_biocam and the gap fill import ok")
PY

cat <<MSG

  environment ready:  $ENV_TARGET   ($ACTIVE)
  repository:         $REPO

  The other steps find the repo on their own, via $HERE/.mea_repo -- nothing
  to export.

  activate:   conda activate $ENV_TARGET
  next: in a project folder, write samples.tsv and mea.env, then submit from
  a login node with $HERE/submit_samples.sh (README.md).

MSG
