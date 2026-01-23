```bash
```
### 📂 Repository Structure
```
.
├── __pycache__/
├── .venv/
│
├── Best_paths/
├── CNN_initial_saved_pytorch_model_weights/
│
├── Figs/
├── JSON_logs/
├── Model_evaluation_figs/
│
├── Pre-obtained_data/
│   ├── dataset_homo_small.mat
│   ├── DTOFs_Homo_raw.csv
│   └── DTOFs_Homo_labels.csv
│
├── DL_CNN_initial.ipynb
├── DL_Full_Pipeline.ipynb
├── DL_PostProcessing.ipynb
│
├── DTOF_plot.py
├── DTOF_std_plot.py
│
├── DTOFs_whiteMC(in).csv
│
├── training_core.py
└── README.md
```

🧭 Directory and File Descriptions
📁 Pre-obtained data/

Contains all input datasets used by the deep-learning pipeline.

dataset_homo_small.mat
Primary dataset used in the current pipeline.
MATLAB v7.3 (HDF5) file containing:

```
X  # DTOFs (N, T)
y  # optical property labels [μa, μs′]
t  # time vector
```

DTOFs (X)

optical property labels (y = [μa, μs′])

time vector (t)

DTOFs_Homo_raw.csv, DTOFs_Homo_labels.csv
Legacy CSV-based DTOF and label files retained for reference and comparison.

📓 Jupyter Notebooks

DL_CNN_initial.ipynb
Baseline implementation of the CNN inversion framework.
Includes dataset loading, preprocessing, model definition, training, and validation.

DL_Full_Pipeline.ipynb
Extended pipeline including experiment logging, checkpointing, and full evaluation.

DL_PostProcessing.ipynb
Analysis and visualisation of trained model outputs, including error metrics and plots.

🧠 Model & Training Code

training_core.py
Core reusable Python module containing:

DTOFDataset definition

CNN architecture (Net)

training and validation loops

evaluation utilities

CNN_initial_saved_pytorch_model_weights/
Saved PyTorch checkpoints (.pt) storing trained model weights.

📊 Visualisation & Evaluation

Figs/
Training and validation loss curves.

Model evaluation figs/
Prediction vs ground-truth plots, RMSE/MAE summaries, and diagnostic figures.

DTOF_plot.py, DTOF_std_plot.py
Utility scripts for inspecting DTOFs and preprocessing effects.

🗂️ Logging & Reproducibility

JSON logs/
Structured experiment logs capturing:

preprocessing configurations

training hyperparameters

evaluation metrics

Best paths/
Stores selected best-performing configurations or optimisation results.

⚙️ Environment & Metadata

.venv/
Local Python virtual environment used for dependency isolation.

__pycache__/
Auto-generated Python bytecode (safe to ignore).


📘 DTOF Deep Learning Pipeline for Optical Property Inversion

This repository implements a complete, reproducible deep-learning framework for estimating absorption (μa) and reduced scattering (μs′) from Monte Carlo–simulated DTOFs.

🔍 Project Overview

Time-Domain Near-Infrared Spectroscopy (TD-NIRS) captures Distribution of Time-of-Flight (DTOF) curves that encode tissue optical properties.
This project builds a CNN-based inversion model trained on MCX-simulated DTOFs to recover underlying optical properties.

The pipeline includes:

Full data preprocessing and normalisation

Multi-channel DTOF construction (raw, temporal masks, hybrid)

A flexible CNN architecture with auto-detected flattening dimension

A complete training loop with validation, checkpointing, and GPU support

An evaluation module providing MAE / RMSE metrics

A structured instruction manual describing reproducible usage

🧱 Core System Components
1. DTOFDataset

Handles the full preprocessing workflow:

Load DTOFs from CSV

Extract (μa, μs′) labels from column headers

Apply Savitzky–Golay filtering

Clip negative floating-point noise

Standardise each DTOF to zero mean and unit variance

Construct 1, 3, or 4 input channels via:

Raw DTOF

Early/Mid/Late temporal masks

Combined hybrid features

Output per sample:

signal → (C, T)   # channels × time samples  
target → (μa, μs′)

2. CNN Architecture

A domain-inspired 1D convolutional network consisting of:

Three Conv1d → BatchNorm → ReLU → MaxPool blocks

Automatic flatten-size detection via dummy forward pass

Fully connected regressor head producing:

[μa, μs′]

The architecture supports variable input channels (1, 3, or 4).

3. Training Infrastructure

Features:

PyTorch training loop

Train/validation dataloaders

MSE loss over (μa, μs′)

Adam optimiser

GPU/CPU device selection

Best-model checkpointing (best_dtof_cnn.pth)

Loss curve logging and plotting

Output of epoch-wise training + validation losses

4. Evaluation Module

The ModelEvaluator collects:

Prediction vectors across validation set

Ground-truth labels

MAE for μa and μs′

RMSE for μa and μs′

Optional sample-prediction previews

MAPE is computed internally but not used due to instability near small μa values.


📓 DL_CNN_initial.ipynb — Deep Learning Inversion Pipeline

This notebook implements the end-to-end deep learning pipeline for inverting Monte Carlo–simulated DTOFs into tissue optical properties: absorption (μa) and reduced scattering (μs′).

It provides a fully reproducible workflow covering data loading, preprocessing, model training, validation, and evaluation, and serves as the reference implementation for the CNN-based inversion framework used throughout this project.

🔍 Overview

Time-Domain Near-Infrared Spectroscopy (TD-NIRS) produces Distribution of Time-of-Flight (DTOF) curves that encode information about tissue optical properties across photon pathlengths.

In this notebook, a 1D convolutional neural network (CNN) is trained on MCX-simulated DTOFs to recover the underlying optical parameters.
The design explicitly incorporates temporal sensitivity (early / mid / late photons) and dynamic-range stabilisation via logarithmic transforms.

🧱 Core Components
1. DTOFDataset

The DTOFDataset class encapsulates the full preprocessing pipeline and ensures consistent, reproducible data handling.

Data source

MATLAB v7.3 (.mat) files loaded via h5py

Required variables:

X: DTOFs (N, T)

y: labels (μa, μs′)

t: time vector (seconds, converted internally to ns)

Preprocessing steps

Convert time axis from seconds → nanoseconds

Crop DTOFs to a fixed temporal window

Apply Savitzky–Golay smoothing

Clip numerical noise and negative values

Construct multiple input representations:

raw reflectance

log-transformed reflectance

optional raw + log concatenation

Channel construction modes

single: full DTOF

early_mid_late: three temporally gated channels

hybrid_4ch: full DTOF + early / mid / late masks

Output per sample

signal → (C, T)   # channels × time samples
label  → (μa, μs′)

2. CNN Architecture

The inversion model is a domain-inspired 1D CNN designed for long temporal signals.

Architecture

Three convolutional blocks:

Conv1d → BatchNorm → ReLU → MaxPool

Automatic flatten-dimension detection via a dummy forward pass

Fully connected regression head producing:

[log(μa), log(μs′)]


Key properties

Supports variable input channel counts (1, 3, 4, or 8)

No activation on final layer (required for log-space regression)

Input length inferred dynamically from dataset configuration

3. Training Infrastructure

The notebook implements a complete and robust training loop using PyTorch.

Features

Train / validation split using dataset indices (no data leakage)

Mini-batch training via DataLoader

Mean-Squared Error loss in log-parameter space

Adam optimiser

Automatic CPU / GPU device selection

Best-model checkpointing (.pt state dictionary)

Epoch-wise logging of:

training loss

validation loss

RMSE in original physical units

Training is performed on:

targets = log([μa, μs′])


to stabilise optimisation across disparate parameter scales.

4. Evaluation and Metrics

Model performance is assessed using a dedicated evaluation routine.

Metrics reported

MAE for μa and μs′ (original units)

RMSE for μa and μs′ (original units)

Predictions are exponentiated back from log-space prior to evaluation:

μ̂ = exp(model output)


MAPE is intentionally excluded due to instability for small μa values.
