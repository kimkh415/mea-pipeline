# Shared helpers for the slurm scripts. Sourced, never run.

# --- paths ---
# MEA_SCRATCH: work dir (netscratch, purged); sort binary lives here only
# MEA_FINAL:   archive on lab storage, written by 05_archive.sh
#   $MEA_OUT/<prefix>     sorter output   (default $MEA_SCRATCH/sorted)
#   $MEA_RESULTS/<prefix> mea run output  (default $MEA_SCRATCH/results)
# Defaults below are the developer's; submit_samples.sh requires both roots.
MEA_SCRATCH_DEFAULT="/n/netscratch/arlotta_lab/Users/kwanho/ks4"
MEA_FINAL_DEFAULT="/net/bos-nfsisilon/ifs/rc_labs/arlotta_lab2/Users/kwanho/projects/MEA_pipeline_test/results/test"

mea_paths() {
    export MEA_SCRATCH="${MEA_SCRATCH:-$MEA_SCRATCH_DEFAULT}"
    export MEA_OUT="${MEA_OUT:-$MEA_SCRATCH/sorted}"          # sorter output
    export MEA_RESULTS="${MEA_RESULTS:-$MEA_SCRATCH/results}"  # mea run output
    export MEA_FINAL="${MEA_FINAL:-$MEA_FINAL_DEFAULT}"
}

# --- repo ---
# Scripts may be copied out of the checkout, so the repo is resolved, not assumed two levels up
REPO_URL_DEFAULT="https://github.com/kimkh415/mea-pipeline.git"
# A stale default installs old code silently; a test pins it to the current branch
REPO_REF_DEFAULT="main"

# Directory name a clone of the URL lands in
repo_dirname() {
    local url="${MEA_REPO_URL:-$REPO_URL_DEFAULT}"
    basename "${url%.git}"
}

# Echo the expected repo path: $MEA_REPO, else .mea_repo, else ../..
# MEA_REPO may be the checkout or its parent directory.
repo_candidate() {
    local here="${1:?repo_candidate needs the script directory}" base
    if [[ -n "${MEA_REPO:-}" ]]; then
        base="$MEA_REPO"
    elif [[ -f "$here/.mea_repo" ]]; then
        # Written by 01_setup_env.sh; already resolved
        cat "$here/.mea_repo"; return
    else
        base="$(cd "$here/../.." && pwd)"
    fi
    if ! repo_ok "$base" && repo_ok "$base/$(repo_dirname)"; then
        echo "$base/$(repo_dirname)"
    else
        echo "$base"
    fi
}

repo_ok() { [[ -f "${1:-}/pyproject.toml" ]]; }

# Resolve the repo or exit (every step but setup)
find_repo() {
    local here="${1:?find_repo needs the script directory}" repo
    repo="$(repo_candidate "$here")"
    if ! repo_ok "$repo"; then
        echo "no repository at $repo (no pyproject.toml)." >&2
        echo "Run 01_setup_env.sh first -- it clones the repo and records" >&2
        echo "where it put it.  Or set MEA_REPO to an existing checkout." >&2
        exit 1
    fi
    echo "$repo"
}

# Resolve or clone the repo (setup only); path on stdout, messages on stderr
ensure_repo() {
    local here="${1:?ensure_repo needs the script directory}" repo url ref
    repo="$(repo_candidate "$here")"
    url="${MEA_REPO_URL:-$REPO_URL_DEFAULT}"
    ref="${MEA_REPO_REF:-$REPO_REF_DEFAULT}"

    if repo_ok "$repo"; then
        echo "using the checkout already at $repo" >&2
        # Report the ref; an existing checkout is not updated
        if [[ -d "$repo/.git" ]]; then
            echo "  on $(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"                  "at $(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo '?')" >&2
            echo "  NOT updated automatically; git pull it yourself if you want" >&2
            [[ -n "${MEA_REPO_REF:-}" ]] &&                 echo "  (MEA_REPO_REF=$ref ignored: this checkout already exists)" >&2
        fi
    else
        [[ -n "${MEA_REPO:-}" ]] || repo="$HOME/$(repo_dirname)"

        # Clone target: missing or empty dir -> that path; non-empty dir -> <dir>/<repo name>
        if [[ -e "$repo" ]]; then
            [[ -d "$repo" ]] || {
                echo "$repo is not a directory." >&2; exit 1; }
            if [[ -n "$(ls -A "$repo" 2>/dev/null)" ]]; then
                repo="$repo/$(repo_dirname)"
                echo "treating $(dirname "$repo") as a parent directory" >&2
                [[ -e "$repo" ]] && {
                    echo "$repo exists but has no pyproject.toml." >&2
                    echo "Move it aside, or point MEA_REPO at a real checkout." >&2
                    exit 1; }
            fi
        fi
        echo "no checkout found -- cloning $url ($ref)" >&2
        echo "  into $repo" >&2
        command -v git >/dev/null || { echo "git is not on PATH" >&2; exit 1; }
        mkdir -p "$(dirname "$repo")"
        git clone --branch "$ref" --single-branch "$url" "$repo" >&2 || {
            echo "clone failed. A compute node reaches the internet through a" >&2
            echo "proxy; if pip worked and git did not, try cloning on a login" >&2
            echo "node and re-running with MEA_REPO=<path>." >&2
            exit 1; }
        repo_ok "$repo" || { echo "cloned, but $repo has no pyproject.toml" >&2; exit 1; }
    fi

    # Canonicalise before recording
    repo="$(cd "$repo" && pwd -P)"

    # Record beside the scripts so later steps need no MEA_REPO
    if ! echo "$repo" > "$here/.mea_repo" 2>/dev/null; then
        echo "  (could not write $here/.mea_repo -- it is read-only." >&2
        echo "   export MEA_REPO=$repo before the other steps)" >&2
    fi
    echo "$repo"
}

# --- conda ---
# Setup finds or installs conda; other steps only activate. Shell profile untouched
# unless MEA_CONDA_INIT=1.
CONDA_HOME_DEFAULT="$HOME/miniforge3"
MINIFORGE_URL_DEFAULT="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname -s)-$(uname -m).sh"

# Env to use: $MEA_CONDA_ENV (name, or path if it has a /), else .mea_env, else MEA.
# MEA_ENV_PREFIX / MEA_ENV_NAME are older spellings.
env_target() {
    local here="${HERE:-${MEA_SCRIPTS:-}}"
    if [[ -n "${MEA_CONDA_ENV:-}" ]]; then echo "$MEA_CONDA_ENV"
    elif [[ -n "${MEA_ENV_PREFIX:-}" ]]; then echo "$MEA_ENV_PREFIX"
    elif [[ -n "${MEA_ENV_NAME:-}" ]]; then echo "$MEA_ENV_NAME"
    elif [[ -n "$here" && -s "$here/.mea_env" ]]; then cat "$here/.mea_env"
    else echo "MEA"
    fi
}

# -p for a path, -n for a name
env_flag() {
    if [[ "$(env_target)" == */* ]]; then printf '%s\n' -p; else printf '%s\n' -n; fi
}

# Source a conda install's hook; return 1 if absent
_try_conda_hook() {
    local base="${1:-}"
    [[ -n "$base" && -f "$base/etc/profile.d/conda.sh" ]] || return 1
    # shellcheck disable=SC1090
    source "$base/etc/profile.d/conda.sh"
    return 0
}

# Find conda without installing; return 1 if none
find_conda() {
    local base
    _try_conda_hook "${MEA_CONDA_HOME:-}" && return 0
    if command -v conda >/dev/null 2>&1; then
        base="$(conda info --base 2>/dev/null || true)"
        _try_conda_hook "$base" && return 0
    fi
    for base in "$CONDA_HOME_DEFAULT" "$HOME/miniconda3" "$HOME/anaconda3"; do
        _try_conda_hook "$base" && return 0
    done
    # Try Lmod modules (names FAS RC has used)
    if command -v module >/dev/null 2>&1; then
        local m
        for m in Mambaforge Miniforge3 python Anaconda3; do
            module load "$m" >/dev/null 2>&1 || continue
            if command -v conda >/dev/null 2>&1; then
                base="$(conda info --base 2>/dev/null || true)"
                _try_conda_hook "$base" && return 0
            fi
            module unload "$m" >/dev/null 2>&1 || true
        done
    fi
    return 1
}

# Find conda or install Miniforge (setup only)
ensure_conda() {
    local home url
    if find_conda; then
        echo "conda: $(conda info --base)" >&2
        return 0
    fi

    home="${MEA_CONDA_HOME:-$CONDA_HOME_DEFAULT}"
    url="${MEA_MINIFORGE_URL:-$MINIFORGE_URL_DEFAULT}"
    echo "no conda found (not on PATH, not in \$HOME, no module provides it)" >&2
    echo "installing Miniforge to $home" >&2
    echo "  from $url" >&2
    echo "  conda-forge's official installer; set MEA_CONDA_HOME to move it," >&2
    echo "  or install conda yourself and re-run -- this step is skipped when" >&2
    echo "  conda is already reachable." >&2

    if [[ -e "$home" ]]; then
        echo "$home exists but has no etc/profile.d/conda.sh." >&2
        echo "Move it aside, or point MEA_CONDA_HOME at a working install." >&2
        exit 1
    fi

    local tmp; tmp="$(mktemp -d)"
    local sh="$tmp/miniforge.sh"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$sh"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$sh" "$url"
    else
        echo "neither curl nor wget is available to fetch the installer" >&2
        exit 1
    fi
    # A proxy error page can still exit 0; check for a shebang
    [[ -s "$sh" ]] && head -c2 "$sh" | grep -q '#!' || {
        echo "the download is not a shell script -- a compute node reaches the" >&2
        echo "internet through a proxy; try this on a login node, or set" >&2
        echo "MEA_CONDA_HOME to an install you made yourself." >&2
        rm -rf "$tmp"; exit 1; }

    bash "$sh" -b -p "$home" >&2          # -b batch, no prompts, no shell init
    rm -rf "$tmp"
    _try_conda_hook "$home" || { echo "installed to $home but no hook there" >&2; exit 1; }
    echo "conda: $(conda info --base)" >&2

    if [[ "${MEA_CONDA_INIT:-0}" == "1" ]]; then
        echo "MEA_CONDA_INIT=1: running conda init, which edits your shell profile" >&2
        conda init >&2 || true
    else
        echo >&2
        echo "  Your shell profile was NOT modified.  These scripts source" >&2
        echo "  conda's hook themselves and do not need it.  For an interactive" >&2
        echo "  shell, either run once:" >&2
        echo "      $home/bin/conda init bash" >&2
        echo "  or add to ~/.bashrc:" >&2
        echo "      source $home/etc/profile.d/conda.sh" >&2
    fi
}

# Activate the env (every step but setup)
activate_env() {
    local target; target="$(env_target)"
    find_conda || {
        echo "conda is not available, and only 01_setup_env.sh installs it." >&2
        echo "Run that first." >&2
        exit 1; }
    conda activate "$target" 2>/dev/null || {
        echo "no conda environment '$target'." >&2
        echo "Run 01_setup_env.sh first, or set MEA_CONDA_ENV to an environment" >&2
        echo 'that exists (a name or a path).  `conda env list` shows them.' >&2
        exit 1; }
}

# Does the env exist? Lets submit scripts fail on the login node, not in a queued job
env_exists() {
    find_conda >/dev/null 2>&1 || return 1
    local t; t="$(env_target)"
    if [[ "$t" == */* ]]; then [[ -x "$t/bin/python" ]]; return; fi
    conda env list | awk '{print $1}' | grep -qx "$t"
}
