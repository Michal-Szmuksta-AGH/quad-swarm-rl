#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Bootstrap a vast.ai instance to run quad-swarm-rl training.
#
# Designed for the official "nvidia/cuda" template (Ubuntu base, CUDA driver
# pre-installed).  Idempotent -- safe to re-run on the same instance.
#
# Installs:
#   * Miniconda3 (latest, into ~/miniconda3)
#   * Conda env  : swarm-rl  (Python 3.11.10)
#   * PyTorch    : 2.5.0 + CUDA 12.4 wheel
#   * swarm_rl   : pip install -e . (uses this repo's setup.py)
#
# Run from the repository root after rsync upload:
#   cd ~/quad-swarm-rl
#   bash vast_setup.sh
#
# Reactivate the env later (in a new shell):
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate swarm-rl
# ---------------------------------------------------------------------------
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="swarm-rl"
PY_VERSION="3.11.10"

# PyTorch wheel selection.
#   * cu128 = ships kernels for CC up to sm_120 (Blackwell, RTX 50 series).
#   * cu124 = ships kernels up to sm_90 (Hopper).
# Default to cu128 so the script works on both old (RTX 30/40) and new (RTX 50)
# GPUs.  Override via env var if cu128 wheel is unavailable for a specific
# torch version:  CUDA_TAG=cu126 bash vast_setup.sh
CUDA_TAG="${CUDA_TAG:-cu128}"
# Leaving TORCH_VERSION blank installs the latest version compatible with
# CUDA_TAG.  Pin only if you need to reproduce a specific run.
TORCH_VERSION="${TORCH_VERSION:-}"

echo "=== quad-swarm-rl vast.ai setup ==="
echo "Repo:        $REPO_DIR"
echo "Conda dir:   $CONDA_DIR"
echo "Env:         $ENV_NAME (python $PY_VERSION)"
if [ -n "$TORCH_VERSION" ]; then
  echo "Torch:       $TORCH_VERSION+$CUDA_TAG"
else
  echo "Torch:       latest+$CUDA_TAG"
fi
echo ""

# --- 1. Install miniconda (skip if already present) ------------------------
if [ ! -d "$CONDA_DIR" ]; then
  echo "[1/5] Installing miniconda to $CONDA_DIR ..."
  TMP_INSTALLER=$(mktemp /tmp/miniconda-XXXX.sh)
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$TMP_INSTALLER"
  bash "$TMP_INSTALLER" -b -p "$CONDA_DIR"
  rm "$TMP_INSTALLER"
else
  echo "[1/5] Miniconda already at $CONDA_DIR — skipping."
fi

# Make conda available in this shell
source "$CONDA_DIR/etc/profile.d/conda.sh"

# --- 2. Create the env if missing ------------------------------------------
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[2/5] Creating conda env '$ENV_NAME' (python=$PY_VERSION) ..."
  conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
else
  echo "[2/5] Env '$ENV_NAME' already exists — skipping create."
fi

conda activate "$ENV_NAME"

# --- 3. Install swarm_rl + pinned deps from setup.py FIRST ------------------
# This brings in the CPU/old-CUDA torch pinned in setup.py.  We replace it
# in the next step with a wheel that has the right kernels for our GPU.
echo "[3/5] pip install -e $REPO_DIR ..."
pip install --upgrade pip wheel setuptools
cd "$REPO_DIR"
pip install -e .

# --- 4. Replace torch with a wheel that matches the actual GPU's CC --------
# RTX 50 series (Blackwell, sm_120) needs cu128 + torch >= 2.7.  RTX 30/40
# series (sm_86/89) work with cu124 too, but cu128 also works for them.
# pip install -e .'s torch==2.5.0 pin only applies during dependency
# resolution; once installed, nothing checks it at runtime.
if [ -n "$TORCH_VERSION" ]; then
  TORCH_SPEC="torch==$TORCH_VERSION"
else
  TORCH_SPEC="torch"
fi
echo "[4/5] Replacing torch with $TORCH_SPEC ($CUDA_TAG) for this GPU ..."
pip install --upgrade --force-reinstall "$TORCH_SPEC" \
  --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

# --- 5. Verify ----------------------------------------------------------
echo "[5/5] Verifying installation ..."
python - <<'PY'
import sys, torch, sample_factory
print(f"  python           : {sys.version.split()[0]}")
print(f"  torch            : {torch.__version__}")
print(f"  cuda available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  cuda device      : {torch.cuda.get_device_name(0)}")
    print(f"  cuda capability  : {torch.cuda.get_device_capability(0)}")
    # Smoke test: run a kernel that exercises the GPU's actual sm_XX.
    x = torch.randn(256, 256, device="cuda")
    y = torch.tanh(x) @ x  # touches a few common cublas / elementwise kernels
    torch.cuda.synchronize()
    print(f"  gpu kernel test  : OK")
import gym_art.quadrotor_multi.quadrotor_multi  # noqa: F401
from swarm_rl.train import parse_swarm_cfg, register_swarm_components  # noqa: F401
print("  swarm_rl import  : OK")
PY

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start training (e.g. pilot seed 0):"
echo "  source $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME"
echo "  cd $REPO_DIR"
echo "  NUM_WORKERS=88 SEED=0 EXPERIMENT=paper_baseline_8drones_s0 \\"
echo "    bash train_paper_baseline_8drones.sh 2>&1 | tee train_s0.log"
echo ""
echo "Detach with tmux/screen so the run survives SSH disconnect:"
echo "  tmux new -s train"
echo "  # run training command, then Ctrl+B then D to detach"
echo "  tmux attach -t train  # to reattach"
