# Train Template

This is a template repository for training machine learning models for neural data using PyTorch Lightning. It implements a simple next-token prediction model on the Braintree Bank dataset, allowing for easy customization and extension with a structured setup for data handling, model definition, and training processes. It uses Neural libraries such as [crane](https://github.com/insight-neuro/crane) and [torch brain](https://torchbrain.org/) to facilitate working with neural data and features.

## Quick Start

1. Install the required dependencies (we recommend using [uv](https://pypi.org/project/uv/) for managing virtual environments):

   ```bash
   uv sync  # if not using uv, use: pip install -e .[dev] with a virtual environment
   ```

2. Download and preprocess the [Braintree Bank dataset](https://braintreebank.dev/) (~230 GB) using the provided SLURM script (`scripts/data.sh`) or manually via [brainsets](https://github.com/insight-neuro/brainsets), and update your `.env` file with the path to the dataset. Note you may need to adjust the script and `scripts/env.sh` to fit your cluster setup.

   ```
   ROOT_DIR=/path/to/braintree_bank_dataset
   ```

3. Modify the model, training logic, and configuration files as needed (see below). Especially, you will want to set the `wandb.project` and `wandb.entity` in the configuration to log your training runs to [Weights & Biases](https://wandb.ai/).

4. Run the training script:

   ```bash
   uv run -m train [CLI overrides]  # if not using uv, use: python -m train
   ```

   or if using SLURM:

   ```bash
   sbatch scripts/train.sh [CLI overrides]
   ```

## Repository Structure

- `configs/`: Contains configuration files for different training setups. We use [Hydra](https://hydra.cc/) for configuration management, allowing easy CLI overrides and organization.
- `train/`: Contains the source code for data modules, models, and training scripts.
  - `model.py`: Model architecture.
  - `data_module.py`: PyTorch Lightning DataModule for loading and preprocessing the dataset.
  - `featurizer.py`: Code for feature extraction from raw neural data. Will be applied to each sample in the dataset to create the input features for the model.
  - `pl_module.py`: PyTorch Lightning module that wraps the model and defines training/validation steps.
  - `dataset.py`: Data loading and preprocessing. You can probably ignore this for now.
  - `train.py`: The main training script that orchestrates the training process.
- `scripts/`: Scripts to run on a SLURM cluster. May need to be adjusted to your cluster configuration.
   - `data.sh`: SLURM script to download and prepare the Braintree Bank dataset using `brainsets`.
   - `env.sh`: SLURM script to set up the environment for training jobs, including loading modules and activating virtual environments. Update this to fit your cluster's environment management.
