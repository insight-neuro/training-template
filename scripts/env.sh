#!/usr/bin/env bash
# Build the environement used for training.
# Designed for the ETH Zurich Euler cluster,
# you might need to adapt it for other systems.

set -euo pipefail

mkdir -p logs

export ROOT_DIR="${SCRATCH:-$(pwd)}/bfm"
export HF_HOME="$ROOT_DIR/.hf/"
export UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv"

module load stack/2024-06 gcc/12.2.0 python/3.12.8 cuda/12.8.0 eth_proxy

if ! command -v uv &> /dev/null ; then
    echo "uv could not be found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"
fi