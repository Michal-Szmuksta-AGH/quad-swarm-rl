#!/usr/bin/env bash
# Bootstrap vast.ai instance for quad-swarm-rl. Run from inside the cloned repo.
# Override CUDA wheel with: CUDA_TAG=cu126 bash vast_setup.sh
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="swarm-rl"
PY_VERSION="3.11.10"
CUDA_TAG="${CUDA_TAG:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-}"

# Miniconda
if [ ! -d "$CONDA_DIR" ]; then
  TMP_INSTALLER=$(mktemp /tmp/miniconda-XXXX.sh)
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$TMP_INSTALLER"
  bash "$TMP_INSTALLER" -b -p "$CONDA_DIR"
  rm "$TMP_INSTALLER"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Env
conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" \
  || conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
conda activate "$ENV_NAME"

# swarm_rl + deps
pip install --upgrade pip wheel setuptools
cd "$REPO_DIR" && pip install -e .

# Replace torch with a wheel matching the actual GPU's CC (Blackwell needs cu128)
TORCH_SPEC="${TORCH_VERSION:+torch==$TORCH_VERSION}"
TORCH_SPEC="${TORCH_SPEC:-torch}"
pip install --upgrade --force-reinstall "$TORCH_SPEC" \
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
