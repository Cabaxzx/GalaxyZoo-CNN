# Galaxy Zoo Morphology Pipeline

A comprehensive PyTorch based AI convolutional Neural Network (CNN) model for predicting galaxy morphology using the Kaggle Galaxy Zoo dataset.

This codebase models galaxy classification as both a discrete categorization task and a continuous probability regression task. Given a telescope image, the CNN  outputs either rigid class labels or the expected fractional crowdsourced consensus votes for specific structural features, while actively enforcing the logical dependencies of the dataset's underlying decision tree.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Data Structure & Decision Tree Logic](#2-data-structure--decision-tree-logic)
3. [System Requirements & Setup](#3-system-requirements--setup)
4. [Repository Structure](#4-repository-structure)
5. [Codebase Architecture & Modules](#5-codebase-architecture--modules)
6. [Methodological Implementations](#6-methodological-implementations)
7. [Usage Guide](#7-usage-guide)
8. [Evaluation Metrics](#8-evaluation-metrics)

---

## 1. Project Overview

The Kaggle Galaxy Zoo dataset consists of thousands of galaxy images labeled by crowdsourced voting fractions across a sequential decision tree. This project is structured to solve three distinct morphological tasks:

### Task 1: Discrete Classification
Targets the primary root nodes of the decision tree, converting probability fractions into hard labels.
* **Binary Classification:** Differentiates strictly between Smooth (elliptical) and Disk galaxies.
* **Multi-Class Classification:** Differentiates between Smooth, Disk, and Camera Artifacts.

### Tasks 2 & 3: Continuous Fractional Regression
Predicts the continuous voting fractions for downstream questions (e.g., "Is it edge-on?", "Is there a ring?"), mapped to 17 specific parameters. The pipeline supports two architectural approaches to this problem:
* **Flat Architecture:** A single linear head predicts all 17 features independently. Structural integrity is enforced post-prediction via a custom hierarchical penalty loss function.
* **Tree Architecture:** The network routes predictions through internal Bayesian nodes. Features are predicted by chaining conditional probabilities down the branches, guaranteeing mathematical compliance with the decision tree.

---

## 2. Data Structure & Decision Tree Logic

The target variables in this dataset are not standard one-hot encoded categories, but rather continuous fractional probabilities derived from the votes of hundreds of thousands of citizen scientists. 

### The Hierarchical Decision Tree
The voting process followed a strict logical hierarchy. For instance:
* **Question 1 (Root Node):** Is the galaxy Smooth, a Disk, or an Artifact?
* **Question 2 (Child Node):** *Only if it is a Disk*, could it be viewed edge-on?
* **Question 7 (Child Node):** *Only if it is Smooth*, how round is it?

### Marginal vs. Conditional Probabilities
Kaggle provides the marginal probabilities—the raw percentage of total users who voted for a specific end-feature. However, a mathematically rigorous decision tree requires conditional probabilities (e.g., the probability it is edge-on given that it is a disk).

To support the "Tree Architecture" approach, the dataset loaders actively reconstruct the underlying conditional probabilities by dividing child feature votes by the sum of their parent nodes. The model predicts these strict conditional distributions natively via Softmax, and dynamically multiplies them back down the tree during the forward pass to output the final Kaggle-compliant marginal fractions.

---

## 3. System Requirements & Setup

### PyTorch Compatibility
The codebase relies entirely on standard PyTorch 2.x functions (such as `torch.amp.autocast` for mixed precision). 
* **Standard Hardware:** If you are using a standard NVIDIA GPU (e.g., RTX 30-series or 40-series) or CPU, any stable release of PyTorch will work perfectly.
* **Next-Gen Hardware:** If you are using an Ada Lovelace or Blackwell architecture GPU (e.g., RTX 5060), you must install a PyTorch Nightly build, as current stable releases lack the necessary architecture flags to maintain FP16 scaling stability on these specific cards.

### Environment Specifications
* **OS:** Windows 11 / Linux (Ubuntu 22.04+)
* **Python:** `3.12+`

### Installation
Ensure at least Python 3.12 is installed, then configure the virtual environment and install dependencies using: 
```bash
uv sync
```
Alternatively, use pip:

```bash
# 1. Install PyTorch (Use stable for normal GPUs, or nightly for RTX 50-series)
pip install torch torchvision
# OR for RTX 50-Series: pip install --pre torch torchvision --index-url [https://download.pytorch.org/whl/nightly/cu121](https://download.pytorch.org/whl/nightly/cu121)

# 2. Install data processing, API, and visualization dependencies
pip install numpy pandas scikit-learn matplotlib seaborn tqdm pillow kagglehub ipywidgets
```

---

## 4. Repository Structure

```text
├── data/                      # Auto-generated by data_loading.ipynb
│   ├── images/                # Directory for raw galaxy JPEG images
│   ├── labels.csv             # Ground-truth targets and probability fractions
│   └── dataset_stats.json     # Automatically generated cache for dataset mean/std
├── utils/
│   ├── __init__.py
│   ├── data.py                # Dataloader instantiation, augmentation, and sampling
│   ├── model.py               # CNN architecture, residual blocks, and routing heads
│   ├── train.py               # Optimization loops and custom hierarchical loss functions
│   └── plotting.py            # Inference logic, Test-Time Augmentation, and plotting
├── data_loading.ipynb         # Initialization notebook to fetch Kaggle datasets
├── classification_run.ipynb   # Main execution notebook for Task 1
└── regression_run.ipynb       # Main execution notebook for Tasks 2 & 3
```

---

## 5. Codebase Architecture & Modules

The repository is modularized to strictly separate data processing, model geometry, optimization, and evaluation.

### `utils/data.py` (Data Pipeline)
Handles dataset instantiation, continuous target extraction, and tensor formatting.
* **`ClassificationDataset` / `RegressionDataset`:** Custom PyTorch Dataset classes. The regression dataset dynamically extracts Bayesian conditional probabilities from marginal crowdsourced votes when the tree architecture is selected.
* **`norm_params()`:** Iterates over the training set to compute global RGB means and standard deviations. Results are cached to `dataset_stats.json` for reproducibility and loading speed.
* **`transformation()`:** Applies deterministic structural preparation and stochastic spatial augmentations.

### `utils/model.py` (Model Architecture)
Defines the custom convolutional backbone and dynamic output routing.
* **`DoubleConvolutionBlock`:** The foundational feature extractor. Utilizes ResNet-style 1x1 skip connections to aid gradient flow. Biases are disabled in favor of `BatchNorm2d` to handle internal covariate shift.
* **`CNN` Backbone:** The primary network consists of 5 sequential Double Convolution blocks. After each block, a `MaxPool2d` operation progressively halves the spatial dimensions, enforcing translation invariance. An `AdaptiveAvgPool2d(1)` layer subsequently crushes the final feature maps into a dense, globally pooled 512-dimensional vector, regulated by a 35% Dropout layer to prevent memorization.
* **Dynamic Heads:** The 512-d vector is routed based on the `ARCH` parameter. The `flat` mode connects to a single unified linear head. The `tree` mode spawns explicit independent linear layers for discrete node branches (e.g., Q1, Q2, Q7), mapping internal Softmax activations to satisfy conditional probability constraints.

### `utils/train.py` (Training Logic)
Manages the optimization loop, gradient scaling, and loss computation.
* **`fit()`:** The primary training orchestrator. Initializes the `torch.autocast` context for mixed precision, updates weights via `AdamW`, and modulates learning rate using `OneCycleLR`.
* **`FullHierarchicalLoss`:** A custom structural penalty module utilized by the flat regression architecture. It computes standard base loss, applies an internal sigmoid, and calculates an `MSELoss` penalty if a node's independent predictions fail to sum to its parent node's physical probability constraints.

### `utils/plotting.py` (Inference & Analytics)
Handles deterministic validation and statistical visualization.
* **`find_optimal_thresholds()`:** Utilizes a vectorized grid search to dynamically calculate decision boundaries that maximize the Macro F1-score.
* **`evaluate_model()`:** Implements Test-Time Augmentation (TTA). Passes raw, horizontally flipped, and vertically flipped views through the network, averaging predictions to smooth spatial variance.

---

## 6. Methodological Implementations

To ensure experimental reproducibility and mathematical stability, the following protocols are embedded in the pipeline:

**Data Preparation & Augmentation:**
* **Center Cropping:** The raw images contain significant empty black space and static telescope border artifacts. A strict 230px `CenterCrop` is applied uniformly before resizing.
* **Orientation Invariance:** In astrophysics, galaxies have no universal "up" or "down". The training pipeline leverages this by randomly applying `HorizontalFlip`, `VerticalFlip`, and continuous 180-degree `RandomRotation`.
* **Color Jittering:** Telescope sensor exposure variations are simulated using a 15% brightness and contrast jitter.

**Inverse Frequency Sampling:**
Morphologies like rings or gravitational lenses are statistically rare. For regression models, an inverse-frequency continuous sampler (`WeightedRandomSampler`) guarantees minority features appear uniformly during epoch batches.

**Target Softening:**
To prevent gradients from exploding when predicting `0.0` or `1.0` probabilities, the data loader linearly compresses all regression ground-truths to a `[0.01, 0.99]` distribution.

---

## 7. Usage Guide

*Note: All Jupyter notebooks must be executed from the root directory of the repository so the Python interpreter correctly registers the local `utils/` module imports.*

### Phase 1: Data Acquisition
Before running any models, you must fetch the dataset via the Kaggle API. Open `data_loading.ipynb`.
1. Replace the `KAGGLE_API_TOKEN` string with your personal Kaggle authentication token. *(Warning: Do not commit your token to version control).*
2. Execute the notebook. It will automatically download the 850MB dataset, extract the images, format the target CSV, and build the `data/` directory structure required by the dataloaders.

### Phase 2: Classification Task
Open `classification_run.ipynb`. Define the target parameters in the configuration block.

```python
from utils.data import classification_dataloader
from utils.model import CNN

# ARCH Options: "classification" (3-class) or "binary" (2-class)
ARCH = "classification" 
IMG_SIZE = 128
BATCH_SIZE = 64

# Tolerance (0.65) filters out highly ambiguous images
train_loader, val_loader, test_loader, train_data = classification_dataloader(
    labels_path, img_size=IMG_SIZE, batch_size=BATCH_SIZE, task=ARCH, tolerance=0.65
)
```

### Phase 3: Regression Task
Open `regression_run.ipynb`. Select between the standard flat baseline or the structurally enforced Bayesian tree.
When running the flat architecture, ensure final activations correspond to the selected loss function (e.g., apply a sigmoid on raw outputs for MSELoss, or evaluate raw logits for BCEWithLogitsLoss).

```python
from utils.data import regression_dataloaders
from utils.train import FullHierarchicalLoss
from torch import nn

# ARCH Options: "flat" or "tree"
ARCH = "flat" 
IMG_SIZE = 200

train_loader, val_loader, test_loader = regression_dataloaders(
    labels_path, img_size=IMG_SIZE, batch_size=64, architecture=ARCH
)

# Route loss dynamically based on architecture constraints
loss_fn = FullHierarchicalLoss(lambda_reg=0.1) if ARCH == "flat" else nn.MSELoss()
```

---

## 8. Evaluation Metrics

Post-training, the `execute_final_evaluation()` function in the Jupyter notebooks triggers a standardized reporting pipeline:

**Classification Metrics:**
* **ROC Curves:** Maps the True Positive Rate against the False Positive Rate. Uses One-vs-Rest (OvR) binarization for 3-class layouts.
* **Normalized Confusion Matrices:** Standardizes row outputs to evaluate class-specific accuracy independently of dataset volumetric bias.

**Regression Metrics:**
* **1D Overlap Histograms:** Tracks the Coefficient of Determination (R²). Visually overlaps true dataset distributions against model prediction distributions.
* **2D Density Heatmaps:** Plots predicted variables against ground truth coordinates. Uses Logarithmic Normalization (`LogNorm`) to preserve visual integrity and contrast for highly sparse occurrences.