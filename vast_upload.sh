#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Upload this quad-swarm-rl directory to a vast.ai instance via rsync.
#
# Run from your LOCAL machine, from inside the quad-swarm-rl/ folder.
#
# Excludes things that don't need to travel:
#   .git/         (the bulk of repo size, not needed to train)
#   train_dir/    (existing checkpoints stay local)
#   cache/        (local cache)
#   __pycache__/  (compiled bytecode)
#   *.egg-info/   (pip metadata)
#   .vscode/      (IDE state)
#
# Usage:
#   bash vast_upload.sh root@ssh4.vast.ai 12345
#       └────┬─────┘ └─────┬──────┘ └─┬──┘
#            user      hostname       port
#
# Get the SSH host + port from vast.ai instance page -> "SSH Direct" section.
# ---------------------------------------------------------------------------
set -e

if [ $# -lt 2 ]; then
  echo "Usage: bash vast_upload.sh <user@host> <port> [remote_path]"
  echo "Example: bash vast_upload.sh root@ssh4.vast.ai 12345"
  echo "Default remote path is ~/quad-swarm-rl"
  exit 1
fi

REMOTE="$1"
PORT="$2"
REMOTE_PATH="${3:-~/quad-swarm-rl}"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Uploading from: $LOCAL_DIR"
echo "       to    : $REMOTE:$REMOTE_PATH"
echo "       port  : $PORT"
echo ""

# --partial      -- resume interrupted transfers
# --progress     -- show progress per file
# --human-readable -- KB / MB / GB
# -avz           -- archive, verbose, compress
rsync -avz --partial --progress --human-readable \
  -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new" \
  --exclude='.git' \
  --exclude='train_dir' \
  --exclude='cache' \
  --exclude='__pycache__' \
  --exclude='*.egg-info' \
  --exclude='.vscode' \
  --exclude='.pytest_cache' \
  --exclude='*.pyc' \
  "$LOCAL_DIR/" \
  "$REMOTE:$REMOTE_PATH/"

echo ""
echo "Upload complete."
echo ""
echo "Next step: SSH into the instance and run setup:"
echo "  ssh -p $PORT $REMOTE"
echo "  cd $REMOTE_PATH && bash vast_setup.sh"
