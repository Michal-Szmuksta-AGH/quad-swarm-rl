#!/usr/bin/env bash
# Download the entire train_dir from a vast.ai instance to local train_dir/.
# Usage: bash vast_download.sh <user@host> <port>
set -e

if [ $# -lt 2 ]; then
  echo "Usage: bash vast_download.sh <user@host> <port>" >&2
  exit 1
fi

REMOTE="$1"
PORT="$2"
REMOTE_PATH="${REMOTE_PATH:-~/quad-swarm-rl/train_dir}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_dir"

mkdir -p "$LOCAL_DIR"

rsync -avz --partial --progress --human-readable \
  -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new" \
  "$REMOTE:$REMOTE_PATH/" \
  "$LOCAL_DIR/"

echo ""
echo "Downloaded to: $LOCAL_DIR/"
