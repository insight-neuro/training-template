# Train Template

This is a template repository for training machine learning models for neural data using PyTorch Lightning. It implements a simple next-token prediction model on the Braintree Bank dataset, allowing for easy customization and extension with a structured setup for data handling, model definition, and training processes.

## Quick Start

1. Install the required dependencies (we recommend using [uv](https://pypi.org/project/uv/) for managing virtual environments):

   ```bash
   uv sync  # if not using uv, use: pip install -e .[dev] with a virtual environment
   ```

2. Download the [Braintree Bank dataset](https://braintreebank.dev/) (~130 GB) using the provided SLURM script (`scripts/braintreebank.sh`) or manually, and update your `.env` file with the path to the dataset:

   ```
   DATA_ROOT_DIR=/path/to/braintree_bank_dataset
   ```

3. Modify the model, train logic, and configuration files as needed (see below).

4. Run the training script:

   ```bash
   uv run -m train  # if not using uv, use: python -m train
   ```

## Repository Structure

- `configs/`: Contains configuration files for different training setups. We use [Hydra](https://hydra.cc/) for configuration management, allowing easy CLI overrides and organization.
- `train/`: Contains the source code for data modules, models, and training scripts.
  - `model.py`: Model architecture.
  - `pl_module.py`: PyTorch Lightning module that wraps the model and defines training/validation steps.
  - `dataset.py`: Data loading and preprocessing. You can probably ignore this for now.
  - `train.py`: The main training script that orchestrates the training process.
- `scripts/`: Scripts to run on a SLURM cluster. May need to be adjusted to your cluster configuration.
   - `braintreebank.sh`: SLURM script to download and prepare the Braintree Bank dataset.