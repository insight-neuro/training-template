#!/usr/bin/env bash
# Script to download and extract the Brain Treebank dataset.
# Run as: sbatch scripts/data.sh

#SBATCH --job-name=braintreebank
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

source scripts/env.sh

DATASET=wang_barbu_braintreebank_2023

RAW_DIR="$ROOT_DIR/braintreebank_raw"
OUT_DIR="$ROOT_DIR/braintreebank"

IEEG_DATA=git+https://github.com/insight-neuro/ieeg-data

echo "Downloading and processing Brain Treebank dataset at $(date)"

# Use ieeg-data to download
uvx --from "$IEEG_DATA" ieeg-data download "$DATASET" \
    --raw-dir "$RAW_DIR" --num_procs 8

echo "Download completed at $(date). Beginning processing."

export UV_NO_CONFIG=1

# Use ieeg-data to process data
uvx --from "$IEEG_DATA" ieeg-data prepare "$DATASET" \
    --raw-dir "$RAW_DIR" --processed-dir "$OUT_DIR"

echo "Processed data saved to $OUT_DIR at $(date). Removing raw data."

# Delete raw data to save space
rm -rf "$RAW_DIR"

echo "Brain Treebank dataset preparation completed at $(date)."
echo "Data available at $OUT_DIR."
