#!/usr/bin/env bash
# Script to download and extract the Braintree Bank dataset.
# Run as: sbatch scripts/braintreebank.sh [ROOT_DIR]

#SBATCH --job-name=braintreebank
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.error
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

set -euo pipefail

mkdir -p logs

module load eth_proxy  # Replace this if not on ETH Zurich cluster

# Optional root dir (default: SCRATCH if set, otherwise cwd)
ROOT_DIR="${1:-${SCRATCH:-$(pwd)}}"

BASE_URL="https://braintreebank.dev/data"
ZIP_DIR="$ROOT_DIR/braintreebank_temp"
OUT_DIR="$ROOT_DIR/braintreebank"

mkdir -p "$ZIP_DIR" "$OUT_DIR"

echo "=== Root dir: $ROOT_DIR ==="
echo "=== Downloading files ==="

wget -c -r -np -nH --cut-dirs=1 \
     -A "*.json,*.zip" \
    -P "$ZIP_DIR" \
    "$BASE_URL"

echo "=== Extracting zip files ==="
find "$ZIP_DIR" -name "*.zip" -print0 \
  | xargs -0 -r -P 8 unzip -oq -d "$OUT_DIR"

echo "=== Moving non-zip files ==="
find "$ZIP_DIR" -type f ! -name "*.zip" -exec mv {} "$OUT_DIR" \;

echo "=== Cleanup ==="
rm -rf "$ZIP_DIR"

echo "=== DONE ==="
