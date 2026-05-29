<div align="center">

# AT-GAN

### Arbitrary Tabular Generative Adversarial Network

*A Tabular GAN framework for generating synthetic tabular data from arbitrary mixed-type tabular datasets.*

[![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-purple.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![W&B](https://img.shields.io/badge/tracking-Weights%20%26%20Biases-yellow.svg)](https://wandb.ai/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Jns-M/at-gan/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/at-gan)](https://pypi.org/project/at-gan/)

</div>

---

## Table of Contents

1. [Overview](#overview)
1. [Key Features](#key-features)
1. [Installation](#installation)
1. [CLI Usage](#cli-usage)
1. [API Usage](#api-usage)
1. [Configuration Reference](#configuration-reference)
1. [In-Training Evaluation Suite](#in-training-evaluation-suite-1)
1. [Synthetic Data Evaluation (Post-Training)](#synthetic-data-evaluation-post-training-1)

---

## Overview

**at-gan** is a framework for training Generative Adversarial Networks on **arbitrary tabular data**. It is designed to work with *continuous*, *binary*, *discrete count*, and *categorical* features within a single pipeline.

The framework combines a **multi-branch generator** (G), a **PacGAN-style discriminator** (D), an integrated **evaluation 
suite**, and **[Weights & Biases](https://wandb.ai/)** (W&B) sweep orchestration, experiment tracking, and training monitoring + visualization.


> **Goal:** Training a GAN that is capable of producing realistic synthetic tabular data from a given dataset with minimal manual tuning and a transparent, observable training process.

---

## Key Features

### Dynamic, Config-Driven Architectures
- Generator and Discriminator built **entirely from YAML-config**.
- Configurable amount of `layers` and `units`.
- Configurable activations: `relu`, `leaky_relu`, `elu`, or any other activation supported in Keras.
- Configurable `dropout` layers.
- Optional `Batch Normalization` for G.

### Mixed-Type Data Handling
- The `TabularPreprocessor` handles **types of input features**:
  - **Continuous** → `MinMaxScaler(-1, 1)` → `tanh` output branch.
  - **Discrete Count** → `MinMaxScaler(0, 1)` → `sigmoid` output branch.
  - **Binary** → 0/1 and optional β-distributed noise application → `sigmoid` output branch.
  - **Categorical** → One-hot encoding and optional label-preserving smoothing → `softmax` output branch.
- Per-column decimal precision preservation.
- Scalers and encoders are stored and reused for inference.

### GAN Training and Stabilization Techniques
| Technique                           | Controlled by                       | What it does                                                                                    |
|-------------------------------------|-------------------------------------|-------------------------------------------------------------------------------------------------|
| **PacGAN packing**                  | `discriminator.pack_size`           | Concatenates *k* rows into a single D input → fights mode collapse                              |
| **One-sided label smoothing**       | `discriminator.label_smoothing_min` | Real labels sampled from `[min, 1.0]` instead of hard `1.0`                                     |
| **Label flipping**                  | `discriminator.label_flipping`      | Random fraction of real labels flipped to `0` to prevent D overconfidence                       |
| **TTUR**                            | `g_lr` / `d_lr`                     | Different LRs for G and D. Sweeps auto-clamp `d_lr ≤ g_lr`                                      |
| **G:D update ratio**                | `g_updates_per_epoch`               | Multiple G steps per D step to balance the training process                                     |
| **LR Cosine decay + warm restarts** | `lr_cosine_decay`                   | `CosineDecayRestarts` schedule with configurable `alpha` floor for the learning rate of G and D |
| **Adam `beta_1` override**          | `adam_beta_1`                       | Typically lowered from default `0.9` for training stability                                     |
| **Gradient clipping**               | *always-on*                         | `clipnorm=1.0` on both Adam optimizers                                                          |

### In-Training Evaluation Suite
Runs every `eval_frequency` epochs on held-out real samples, logs results to W&B, and saves the **best** checkpoint by error score. See [In-Training Evaluation Suite](#in-training-evaluation-suite-1).

### Experiment Tracking
[Weights & Biases](https://wandb.ai/) integration:
- Per-epoch loss/metric logging via a dedicated `WandbCallback`.
- Training visuals: **correlation heatmaps** + **PCA overlap scatter plots**.
- Local-only mode when `--no-wandb` is set (uses `run_id="offline_run"`).

### Sweeps & Neural Architecture Search
- W&B sweeps for **Neural Architecture Search** (NAS) and **Hyperparameter Optimization**.
- Mechanic to resume existing W&B sweeps (and single runs).

### Synthetic Data Evaluation (Post-Training)
- **Privacy**: Distance to Closest Record (DCR)
- **Statistic Fidelity**: [Synthetic Data Vault](https://github.com/sdv-dev/sdv) (SDV)
- **Utility Retention**: Train on Synthetic, Test on Real (TSTR)

### Usage Modes
- 🖥️ **CLI**: `train`, `sweep`, `generate`, `evaluate`.
- 🐍 **Python API** (`at_gan.api`): `train`, `sweep`, `generate`, `evaluate`.

---

## Installation

**Requirements:** Python `3.10 – 3.12` and dependencies listed in `pyproject.toml`.

### Option A: Standard-Installation from [PyPI](https://pypi.org/project/at-gan/) (recommended)

```shell script
pip install at-gan
```

### Option B: Core-Only Installation from [PyPI](https://pypi.org/project/at-gan-core) (recommended for Docker & GPU-Support)

```shell script
pip install at-gan-core
```
Note: This installation does not include *TensorFlow* in its dependencies, making it ideal for training with GPU-Support enabled, e.g. in Docker containers with preconfigured CUDA/cuDNN environments.

### Option C: Editable install from the [GitHub Repository](https://github.com/Jns-M/at-gan)

1. Clone the [GitHub repository](https://github.com/Jns-M/at-gan)
1. Run the following command:

```shell script
pip install -e .
```


### Verify installation

```shell script
at-gan --help
python -c "import at_gan; print(at_gan.__version__)"
```


### Weights & Biases Login (one-time)

```shell script
wandb login
```


> 💡 You can use this framework without W&B by passing `--no-wandb` (CLI) or `enable_wandb=False` (API).

---

## CLI Usage

```shell script
at-gan --help
```


### `train`: Run or resume a single GAN training run

| Flag                     | Short      | Default    | Description                        |
|--------------------------|------------|------------|------------------------------------|
| `--config`               | `-c`       | *required* | Path to the YAML experiment config |
| `--wandb / --no-wandb`   | `-w / -nw` | `--wandb`  | Toggle W&B tracking                |
| `--export / --no-export` | `-e / -ne` | `--export` | Save `.keras` generator file       |
| `--generate-samples`     | `-g`       | `1000`     | Auto-generate *N* samples post-training |

**Examples:**

```shell script
at-gan train -c configs/config.yaml -w -e -g 5000
```

Note: A run can be resumed via the `resume_run_id` config key. See [Configuration Reference](#configuration-reference).

---

### `sweep`: Run or resume a W&B sweep


| Flag | Short | Description                                      |
|---|---|--------------------------------------------------|
| `--base-config` | `-c` | Baseline experiment config                       |
| `--sweep-config` | `-s` | W&B sweep config (required for new sweeps)       |
| `--count` | `-n` | Max runs this agent will execute                 |
| `--sweep-id` | `-id` | Resume an existing sweep instead of creating one |

```shell script
# Launch a new 50-run sweep
at-gan sweep -c configs/config.yaml -s configs/sweep_config.yaml -n 50

# Resume an existing sweep
at-gan sweep -c configs/config.yaml -id abc123 -n 20
```

---

### `generate`: Generate synthetic samples from a trained generator

| Flag | Short | Description                                    |
|---|---|------------------------------------------------|
| `--config` | `-c` | YAML used during the **original** training run |
| `--run-id` | `-r` | W&B run ID or `"offline_run"`                  |
| `--samples` | `-n` | Number of samples to generate                  |
| `--output` | `-o` | Optional override for CSV output path          |

```shell script
at-gan generate -c configs/config.yaml -r a1b2c3 -n 10000 -o synthetic_data.csv
```

Note: `generate` always loads **`best_generator.keras`**, not the latest.

---

### `evaluate`: Run synthetic data evaluation (post-training)

| Flag          | Short | Description                                           |
|---------------|-------|-------------------------------------------------------|
| `--real`      | `-r`  | Path to the real data CSV                             |
| `--synthetic` | `-s`  | Path to the synthetic data CSV                        |
| `--target`    | `-t`  | Discrete target column for (optional) TSTR evaluation |

```shell script
at-gan evaluate -c real_data.csv -r synthetic_data.csv -t target_column
```

Note: TSTR evaluation is only performed if a discrete feature (i.e. binary or categorical) is specified as the target.

---

## API Usage

The Python API exposes the same primary functions as a CLI, making it easy to integrate into existing projects.

See `examples/api_example.py` and `examples/api_example.ipynb` in the [GitHub Repository](https://github.com/Jns-M/at-gan) for a full API usage example.

> Note: The `train` entry point also accepts a `dict` instead of a path to a YAML file as input.

---

## Configuration Reference

Experiments are driven by **two YAML files**: a base config and a sweep config.

See `configs/config.yaml` and `configs/sweep_config.yaml` in the [GitHub Repository](https://github.com/Jns-M/at-gan) for examples and recommended default values for most datasets.

### Base Config Reference

```yaml
# =============================================================
#  EXPERIMENT META
# =============================================================
experiment_name: "test_experiment"   # also output directory name
resume_run_id:   null                # W&B run id to resume from checkpoint (optional)
seed:            1130                # seeds Python, NumPy, TensorFlow

# =============================================================
#  DATA
# =============================================================
data:
  dataset_path: "datasets/example.csv"
  output_path:  "experiments/"          # run artifacts found in 'output_path/experiment_name/run_id/'

  # Column routing — every column the GAN should learn MUST be listed here
  continuous_cols:     ["age", "heart_rate", "glucose"]
  binary_cols:         ["male", "smoker"]
  discrete_count_cols: ["cigs_per_day"]
  categorical_cols:    ["education"]

  # Preprocessing toggles
  treat_bin_as_cat:    false    # route binary cols through OHE + softmax
  beta_noise:          true     # Apply Beta-distributed noise on binary cols
  smooth_categorical:  true     # Apply label-preserving noise on OHE groups

# =============================================================
#  MODEL
# =============================================================
model:
  latent_dim: 32             

  generator:
    units:        [64, 64]     
    dropout:      0.0
    activation:   "relu"        # relu | leaky_relu | elu | ...
    batch_norm:   true          # BatchNorm after each Dense layer
    # negative_slope: 0.2       # used only when activation == "leaky_relu"

  discriminator:
    units:               [256, 256]
    dropout:             0.2
    activation:          "leaky_relu"
    negative_slope:      0.2
    pack_size:           3      # PacGAN packing factor (1 disables packing)
    label_smoothing_min: 0.9    # e.g. real labels ~ [0.9, 1.0]
    label_flipping:      0.05   # e.g. 5% of real labels flipped to 0 each step

# =============================================================
#  TRAINING
# =============================================================
training:
  device:               "cpu"   # "cpu" or "gpu"
  epochs:               2000
  batch_size:           512
  g_updates_per_epoch:  2       # G steps per D step

  # Optimizers
  adam_beta_1:          0.5     # GAN-stable Adam beta_1
  g_lr:                 0.0002  # G Learning Rate
  d_lr:                 0.0003  # D Learning Rate

  # LR schedule
  lr_cosine_decay:                 true
  lr_cosine_decay_restart_epochs:  2000   # restart every N epochs
  g_lr_decay_alpha:                0.1    # minimum G LR fraction (floor)
  d_lr_decay_alpha:                0.1    # minimum D LR fraction (floor)

  # Evaluation & checkpointing
  checkpoint_frequency: 100     # save "latest" every N epochs
  eval_frequency:       100     # run evaluation suite every N epochs
  test_split_pct:       0.2     # percentage of data to hold out for in-training evaluation
```


### Sweep Config Reference

```yaml
# =============================================================
#  SWEEP STRATEGY & METRICS
# =============================================================
method: bayes              

metric:
  name: Eval/Total_Error     # W&B log key
  goal: minimize

early_terminate:
  type: hyperband            # Kills unpromising runs early to save compute time
  min_iter: 300              # Don't kill any run before e.g. epoch 300
  eta: 3                     # The halving rate for the Hyperband brackets

# =============================================================
# PARAMETERS
# =============================================================
parameters:

  # Sweeps choose from a fixed set of hyperparameter values
  model.latent_dim:
    values: [ 16, 32, 64, 128, 256 ]

  # -----------------------------------------------------------
  #  Generator Architecture
  # -----------------------------------------------------------
  generator.num_hidden_layers:
    values: [ 2, 3, 4 ]
  generator.base_units:
    values: [ 32, 64, 128, 256, 512 ]
  generator.max_units:
    value: 512                         
  generator.architecture_shape:
    values: [ "block", "ascending", "descending" ] 
  
  generator.dropout:
    value: 0.0                                   # e.g. Fixed to 0.0
  generator.activation:
    values: [ 'relu', 'leaky_relu' ]
  generator.batch_norm:
    values: [ true, false ]

  # -----------------------------------------------------------
  #  Discriminator Architecture
  # -----------------------------------------------------------
  discriminator.num_hidden_layers:
    values: [ 2, 3, 4 ]
  discriminator.base_units:
    values: [ 32, 64, 128, 256, 512 ]
  discriminator.max_units:
    value: 512
  discriminator.architecture_shape:
    values: [ "block", "ascending", "descending" ]
  
  discriminator.dropout:
    values: [ 0.0, 0.2, 0.3, 0.5 ]              
  discriminator.activation:
    values: [ 'relu', 'leaky_relu' ]
  discriminator.negative_slope:
    values: [ 0.1, 0.2, 0.3 ]                    
  discriminator.pack_size:
    values: [ 1, 3 ]                             
  discriminator.label_smoothing_min:
    values: [ 0.85, 0.9, 0.95, 1.0 ]             
  discriminator.label_flipping:
    values: [ 0.0, 0.05, 0.1 ]                  

  # -----------------------------------------------------------
  #  Training Loop & Optimizers
  # -----------------------------------------------------------
  training.batch_size:
    values: [ 64, 128, 256, 512 ]
  training.g_updates_per_epoch:
    values: [ 1, 2, 3 ]                          
  training.adam_beta_1:
    values: [ 0.2, 0.5, 0.7, 0.9 ] 
  
  # Learning Rates
  training.g_lr:
    distribution: log_uniform_values
    min: 0.00001                                 
    max: 0.001                                  
  training.d_lr:
    distribution: log_uniform_values
    min: 0.000005                               
    max: 0.0005                      # at-gan ensures d_lr <= g_lr

  # Cosine Decay Warm Restart Parameters
  training.lr_cosine_decay_restart_epochs:
    distribution: int_uniform
    min: 100
    max: 1000
  training.g_lr_decay_alpha:
    distribution: log_uniform_values
    min: 0.01                                    # Decay to 1% of max LR
    max: 1                                       # No decay
  training.d_lr_decay_alpha:
    distribution: log_uniform_values
    min: 0.01
    max: 1
```

---

## In-Training Evaluation Suite

Every `eval_frequency` epochs, `GANCallback` generates synthetic samples and runs an evaluation against the held-out real samples to guide the hyperparameter sweep:

| Metric            | Computation                                                                                                               |
|-------------------|---------------------------------------------------------------------------------------------------------------------------|
| PCA Error         | First Wasserstein distance between real and synthetic data across the first five PCA components                           |
| Adversarial Error | Absolute AUC deviation of a Random Forest classifier trained to distinguish real and synthetic data (`\|AUC - 0.5\| × 2`) |
| **Total Error**   | `sqrt((pca_error² + adv_error²) / 2.0)`                                                                                   |

Raw errors are passed through a squashing function (`1 - exp(-x)`) so components ∈ `[0, 1]`.

### Visual artifacts (auto-logged to W&B)
- **Correlation heatmaps**: real, synthetic, and absolute difference.
- **PCA scatter overlay**: first two principal components of real vs. synthetic.

## Synthetic Data Evaluation (Post-Training)

The `evaluate` command runs a comprehensive benchmark suite that assesses the quality of the synthetic data generated by the GAN:

1. **Privacy (DCR)**: Distance to Closest Record. Measures the minimum Euclidean distance (in standard deviations) between synthetic rows and real training rows. Absence of exact memorization is guaranteed if ``Min. DCR > 0``.
2. **Statistic Fidelity (SDV)**: Uses the [Synthetic Data Vault](https://github.com/sdv-dev/SDV) (`sdmetrics` package) to generate a Quality Report, comparing 1D marginal distributions (Column Shapes) and 2D correlations (Column Pair Trends).
3. **Utility Retention (TSTR)**: Train on Synthetic, Test on Real.
    * Splits real data into `real_train` (80%) and `real_test` (20%).
    * Trains a **TRTR** baseline (`RandomForest`, `GradientBoosting`, `LogisticRegression`) on `real_train` → baseline F1 on `real_test`.
    * Trains **TSTR** models on the entire synthetic set → F1 on the **same** `real_test`.
    * Reports `TSTR_Mean_F1 / TRTR_Mean_F1 × 100` (F1-Score Retention in %).
