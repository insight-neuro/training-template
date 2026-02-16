#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --tmp=16G
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:64g
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL

source scripts/env.sh

echo "Starting training job at $(date)"

uv run -m train "$@"

echo "Training job completed at $(date)"
