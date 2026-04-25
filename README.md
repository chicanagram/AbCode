# AbCode

AbCode is a notebook-driven codebase for building predictive models of antibody developability from sequence-derived features, structure-derived features, and other assay or process covariates such as bioprocessing conditions.

The current workflow is centered on:

1. loading antibody datasets
2. generating sequence and structure representations
3. assembling feature matrices
4. training and evaluating supervised ML models

## Scope

Typical use cases include:

- predicting developability-related properties from `VH` / `VL` sequence information
- combining handcrafted descriptors with protein language model features
- incorporating structural outputs where available
- extending models with external metadata columns such as formulation or bioprocess variables

## Repository Layout

```text
notebooks/
  00_explore_data_distribution.ipynb
  00_explore_data_correlations.ipynb
  01_get_structures_apo_holo.ipynb
  02_get_sequence_encodings.ipynb
  03_train_evaluate_supervised_ML_models.ipynb

src/abcode/
  steps/         # notebook-facing workflow entrypoints
  tools/
    encodings/   # classical, physicochemical, and PLM feature generation
    ml/          # feature matrix assembly, splitting, tuning, training, metrics
    openprotein/ # structure/design integrations
    utils/       # sequence, structure, plotting, and general helpers
  core/          # path and pipeline utilities

project_config/
  feature_registry.py
  variables.py

examples/
  expdata/       # small example datasets
  encodings/     # example generated feature files
  ml/            # example metrics and predictions
```

## Main Workflow

The intended entrypoint is the notebook sequence:

1. [00_explore_data_distribution.ipynb](/Users/charmainechia/Documents/projects/AbCode/notebooks/00_explore_data_distribution.ipynb) and [00_explore_data_correlations.ipynb](/Users/charmainechia/Documents/projects/AbCode/notebooks/00_explore_data_correlations.ipynb) for dataset inspection
2. [01_get_structures_apo_holo.ipynb](/Users/charmainechia/Documents/projects/AbCode/notebooks/01_get_structures_apo_holo.ipynb) for optional structure generation
3. [02_get_sequence_encodings.ipynb](/Users/charmainechia/Documents/projects/AbCode/notebooks/02_get_sequence_encodings.ipynb) for feature extraction
4. [03_train_evaluate_supervised_ML_models.ipynb](/Users/charmainechia/Documents/projects/AbCode/notebooks/03_train_evaluate_supervised_ML_models.ipynb) for supervised learning

The matching step modules live in:

- [get_sequence_encodings.py](/Users/charmainechia/Documents/projects/AbCode/src/abcode/steps/get_sequence_encodings.py)
- [get_structures_apo_holo.py](/Users/charmainechia/Documents/projects/AbCode/src/abcode/steps/get_structures_apo_holo.py)
- [train_evaluate_supervised_ml_models.py](/Users/charmainechia/Documents/projects/AbCode/src/abcode/steps/train_evaluate_supervised_ml_models.py)

## Feature Types

Feature generation is configured in [feature_registry.py](/Users/charmainechia/Documents/projects/AbCode/project_config/feature_registry.py).

Current feature families include:

- classical encodings: `one_hot`, `georgiev`
- physicochemical descriptors: `length`, `aac`, `aaindex1`, `ctdc`, `ctdt`
- protein language model features:
  - `esm2-650m`
  - `esmc-600m`
  - `poet2`

Supported PLM outputs include:

- mean-pooled embeddings
- per-residue embeddings
- PLL features
- LLR features
- SVD-reduced embedding variants

Generated arrays are typically written under dataset-specific encoding directories such as:

- [examples/encodings/opensource](/Users/charmainechia/Documents/projects/AbCode/examples/encodings/opensource)

## ML Workflow

The ML stack in [src/abcode/tools/ml](/Users/charmainechia/Documents/projects/AbCode/src/abcode/tools/ml) handles:

- dataset loading
- feature matrix assembly from saved encodings
- train/test split generation
- model construction
- hyperparameter tuning
- cross-validation metrics
- prediction export
- coefficient extraction for interpretable linear models

Example model outputs are available under:

- [examples/ml/output/opensource](/Users/charmainechia/Documents/projects/AbCode/examples/ml/output/opensource)
- [examples/ml/predictions/opensource](/Users/charmainechia/Documents/projects/AbCode/examples/ml/predictions/opensource)

## Data and Path Configuration

Dataset roots are resolved through [variables.py](/Users/charmainechia/Documents/projects/AbCode/project_config/variables.py), which maps logical names such as `examples` or `biostream-developability-data` to filesystem locations.

This means the notebooks are designed to work across multiple local data repositories by changing a small number of path keys rather than rewriting file paths throughout the code.

## Minimal Setup

This repo does not currently ship with a complete packaging manifest, so the simplest local setup is:

```bash
cd /Users/charmainechia/Documents/projects/AbCode
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Then open the notebooks from the repo root and run them with the environment that has the required scientific Python dependencies installed.

Likely required packages include:

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `xgboost` for some model configurations
- model-specific dependencies for PLM inference or structure workflows

## Notes

- The repo is currently optimized for interactive notebook usage rather than polished package distribution.
- Some structure and external-model workflows depend on local credentials or external services.
- Example artifacts in `examples/` are useful as a reference for expected file naming and output structure.
