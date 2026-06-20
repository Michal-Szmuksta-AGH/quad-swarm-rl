#!/usr/bin/env bash
# Bootstrap vast.ai instance for quad-swarm-rl. Run from inside the cloned repo.
# Reuses any conda already on the template; installs miniconda only if none.
# Override CUDA wheel with: CUDA_TAG=cu126 bash vast_setup.sh
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="swarm-rl"
PY_VERSION="3.11.10"
CUDA_TAG="${CUDA_TAG:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-}"

# Use existing conda if the template already has one (common on vast.ai PyTorch
# templates). Otherwise install miniconda to ~/miniconda3.
if command -v conda >/dev/null 2>&1; then
  CONDA_DIR="$(conda info --base)"
  echo "Using existing conda: $CONDA_DIR"
else
  CONDA_DIR="$HOME/miniconda3"
  echo "Installing miniconda to: $CONDA_DIR"
  TMP_INSTALLER=$(mktemp /tmp/miniconda-XXXX.sh)
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$TMP_INSTALLER"
  bash "$TMP_INSTALLER" -b -p "$CONDA_DIR"
  rm "$TMP_INSTALLER"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Accept anaconda channel ToS (conda >=24.9; no-op on older versions)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Env (with pip explicitly to avoid templates that omit it)
conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" \
  || conda create -y -n "$ENV_NAME" "python=$PY_VERSION" pip
conda activate "$ENV_NAME"

# Use `python -m pip` after activate — avoids PATH ambiguity between system pip
# and env pip that can happen on vast.ai containers.
python -m pip install --upgrade pip wheel setuptools
cd "$REPO_DIR" && python -m pip install -e .

# Replace torch with a wheel matching the actual GPU's CC (Blackwell needs cu128)
TORCH_SPEC="${TORCH_VERSION:+torch==$TORCH_VERSION}"
TORCH_SPEC="${TORCH_SPEC:-torch}"
python -m pip install --upgrade --force-reinstall "$TORCH_SPEC" \
  --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

# Verify
python - <<'PY'
import sys, torch
print(f"python: {sys.version.split()[0]}, torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)} (cc {torch.cuda.get_device_capability(0)})")
    x = torch.randn(256, 256, device="cuda")
    torch.tanh(x) @ x; torch.cuda.synchronize()
    print("gpu kernel test: OK")
import gym_art.quadrotor_multi.quadrotor_multi  # noqa: F401
from swarm_rl.train import parse_swarm_cfg, register_swarm_components  # noqa: F401
print("swarm_rl import: OK")
PY

cat <<EOF

Setup done. To train (run inside tmux/screen so it survives SSH drop):
  conda activate $ENV_NAME
  SEED=0 bash train_perception_limited_8drones.sh 2>&1 | tee train.log
EOF
