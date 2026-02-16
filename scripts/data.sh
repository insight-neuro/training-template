#!/usr/bin/env bash
# Script to download and extract multiple Brainsets.
# Run as: sbatch scripts/data.sh

#SBATCH --job-name=data
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

source scripts/env.sh

export UV_NO_CONFIG=1

RAW_DIR="$ROOT_DIR/raw"
OUT_DIR="$ROOT_DIR"

# Use insight-neuro's fork of brainsets to access the datasets
BRAINSETS=git+https://github.com/insight-neuro/brainsets

echo "Processing Brainsets at $(date)"

# List of datasets to process
DATASETS=(
    wang_barbu_braintreebank_2023
    # add more brainsets here
)

# Use brainsets to download and prepare each dataset
for DATASET in "${DATASETS[@]}"; do

    uvx --from "$BRAINSETS" brainsets prepare "$DATASET" \
        --raw-dir "$RAW_DIR" --processed-dir "$OUT_DIR"

        if [ $? -ne 0 ]; then
        echo "Error processing $DATASET. Exiting."
        exit 1
    fi
done

# Delete raw data to save storage space
rm -rf "$RAW_DIR"
    
echo "Dataset preparation completed at $(date)."
echo "Data available at $OUT_DIR."
