# Binding-Aware Protein Design via Fine-Tuned ProteinMPNN

## Overview

This repository contains the implementation and experiments for fine-tuning an inverse folding model (ProteinMPNN) to optimize protein–protein binding affinity using reward-augmented maximum likelihood (RAML). The project evaluates whether binding-aware sequence design improves affinity while maintaining structural integrity.

## Repository Structure

### `data/`

* `processed_data.csv`: Curated and preprocessed dataset used for training and evaluation.

### `figures/`

* Contains final plots and visualizations summarizing model performance and evaluation results.

### `src/`

* `custom_helpers/`: Utilities for data loading and sequence generation.
* `final_samples/`: Generated sequences used for final evaluation.
* `folded_sequences/`: Structure predictions for generated sequences.
* `notebooks/`:

  * Data preprocessing
  * Training workflow
  * Final evaluation, tables, and plots
* Training scripts:

  * `filtering.py`: Data filtering and preprocessing pipeline.
  * `raml_training.py`: Fine-tuning ProteinMPNN using RAML.

### `environment.yaml`

* Conda environment specification with all required dependencies.

## Usage

1. Set up the environment:

   ```
   conda env create -f environment.yaml
   conda activate <env_name>
   ```

2. Train the model:

   ```
   python src/train_script_raml.py
   ```

3. Refer to notebooks in `src/notebooks/` for detailed workflows, analysis, and figure generation.

## Notes

* The repository focuses on scaffold optimization tasks (e.g., affinity maturation).
* Performance on completely novel protein families is limited.
* Structural validation is performed using ESMFold (not included in this repository)
