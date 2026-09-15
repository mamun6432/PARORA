#!/bin/bash
# Set up (if needed) and launch PARORA's full agent (app.py) locally, with the
# conda environment, Ollama models, and AmberTools discovery all wired up.
# This is the path that reaches every tool -- Docker (deploy.sh) only runs
# server.py's 3-tool subset and never bundles AmberTools/PyMOL.
set -e

cd "$(dirname "$0")"

ENV_NAME="parora"
AMBERTOOLS_ENV="ambertools"

# ── 1. Conda environment ──────────────────────────────────────────────────────
# A plain script invocation doesn't source ~/.zshrc or ~/.bashrc, so `conda`
# is not guaranteed to be the shell function conda init set up interactively
# -- on a machine with more than one conda install (Miniconda + a Homebrew
# cask, say) a bare `conda` here can silently resolve to the wrong one and
# "not find" environments that exist under the other root. Locate a real
# conda.sh and source it explicitly so this script always uses the same
# installation, regardless of how it was invoked.
CONDA_ROOT=""
for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
                 "/opt/homebrew/Caskroom/miniconda/base" "/opt/anaconda3" "/opt/miniconda3"; do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
        CONDA_ROOT="$candidate"
        break
    fi
done

if [ -z "$CONDA_ROOT" ]; then
    echo "Could not find a conda installation. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
echo "Using conda at $CONDA_ROOT"

ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"

if [ -x "$ENV_PREFIX/bin/python" ]; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME' from parora.yml..."
    conda env create -f parora.yml
fi

# From here on, call this environment's own binaries by absolute path rather
# than `conda run -n`/`conda activate` -- on a machine with more than one
# conda root, name-based resolution has proven to silently pick the wrong
# root's envs/ directory even after sourcing the intended conda.sh above.
echo "Syncing Python dependencies from protein-viz-agent/requirements.txt..."
"$ENV_PREFIX/bin/pip" install -q -r protein-viz-agent/requirements.txt

# ── 2. Ollama + models ─────────────────────────────────────────────────────────
if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama not found. Install it first: https://ollama.com/download"
    exit 1
fi

if ! curl -s -o /dev/null --max-time 2 http://localhost:11434; then
    echo "Ollama does not appear to be running. Start it (the desktop app, or 'ollama serve') and re-run this script."
    exit 1
fi

for model in qwen2.5:7b llama3.2; do
    if ! ollama list | awk '{print $1}' | grep -qx "$model" && ! ollama list | awk '{print $1}' | grep -qx "${model}:latest"; then
        echo "Pulling $model..."
        ollama pull "$model"
    fi
done

# ── 3. AmberTools discovery (optional: prepare_structure / build_membrane / simulation, quantum, oniom) ──
# `conda env list` only prints a name for envs under the currently-preferred
# root; an env living under a *different* conda root (as ambertools does on
# this machine) is listed by path only, with no name column -- so match by
# either the name column or the path's own basename.
AMBER_ENV_PATH=$(conda env list | awk -v e="$AMBERTOOLS_ENV" \
    '{n=split($NF,parts,"/")} $1==e || parts[n]==e {print $NF; exit}')
if [ -n "$AMBER_ENV_PATH" ] && [ -x "$AMBER_ENV_PATH/bin/packmol-memgen" ]; then
    export PACKMOL_MEMGEN="$AMBER_ENV_PATH/bin/packmol-memgen"
    echo "AmberTools found -- PACKMOL_MEMGEN=$PACKMOL_MEMGEN"
else
    echo "AmberTools env '$AMBERTOOLS_ENV' not found -- prepare/membrane/simulation/QM tools will report"
    echo "unavailable rather than fail. Set it up with:"
    echo "  conda create -n $AMBERTOOLS_ENV --override-channels -c conda-forge ambertools -y"
fi

# ── 4. PyMOL discovery (optional: render_image) ────────────────────────────────
PYMOL_ENV_PATH=$(conda env list | awk '$1 ~ /pymol/ {print $NF; exit}')
if [ -z "${PYMOL_PYTHON:-}" ] && [ -n "$PYMOL_ENV_PATH" ] && [ -x "$PYMOL_ENV_PATH/bin/python" ]; then
    if "$PYMOL_ENV_PATH/bin/python" -c "import pymol2" >/dev/null 2>&1; then
        export PYMOL_PYTHON="$PYMOL_ENV_PATH/bin/python"
        echo "PyMOL found -- PYMOL_PYTHON=$PYMOL_PYTHON"
    fi
fi

# ── 5. Run the full agent ──────────────────────────────────────────────────────
cd protein-viz-agent
mkdir -p structures membranes prepared
exec "$ENV_PREFIX/bin/streamlit" run app.py --server.headless=true
