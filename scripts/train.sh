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

module load stack/2024-06 gcc/12.2.0 python/3.13.0 cuda/12.8.0 eth_proxy

if ! command -v uv &> /dev/null then
    echo "uv could not be found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "Starting training job at $(date)"

uv run -m train "$@"

echo "Training job completed at $(date)"
