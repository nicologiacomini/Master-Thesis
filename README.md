# Master's Thesis: Metadata Capture, Knowledge Extraction, and Predictive Analysis for HPC Workflows 

This repository contains the research and experiments from my Master's thesis at Universitat Politecnica de Catalunya (UPC), developed in collaboration with the Barcelona Supercomputing Center (BSC).

In this repository I show only the last part of the development, which is the research of the best machine learning models to use for execution time prediction. Indeed, the thesis consists of 3 main parts:
- **[Profiling tool implementation](https://github.com/bsc-wdc/compss/tree/stable/compss/runtime/scripts/system/profiling)**: Developing and implementation of the profiling tool for obtaining the resource usage during the execution and standardization of the metadata using RO-Crate specifications.
- **[Improving of the architecture of Provenance Storage](https://github.com/crs4/provenance-storage/pull/17)**: contribution to Provenance Storage project to improve the architecture of the system, implementing a API middleware that decouples the CLI and makes the database interoperable with other applications and accessible remotely. 
- **Research of the best ML models for execution time prediction**: Explained in this repository.

## Machine Learning Models for COMPSs Execution Time Prediction

This repository contains ready-to-use machine learning models for predicting COMPSs application execution times. Each test represents a different approach to modeling and includes both FFNN (Feed-Forward Neural Network) and XGBoost implementations.

## Repository Structure

The repository is organized into 4 test scenarios, each comparing results between original and filtered datasets:

```
PUBLIC/
├── test_1/                          # Test 1: Input size-based prediction
│   ├── original_dataset/
│   │   ├── ffnn/
│   │   └── xgboost/
│   ├── filtered_dataset/
│   │   ├── ffnn/
│   │   └── xgboost/
│   ├── run_test.py                  # Example script to run predictions
│   └── README.md
├── test_2/                          # Test 2: Matrix parameters-based prediction
│   ├── original_dataset/
│   │   ├── ffnn/
│   │   └── xgboost/
│   ├── filtered_dataset/
│   │   ├── ffnn/
│   │   └── xgboost/
│   ├── run_test.py
│   └── README.md
├── test_3/                          # Test 3: USL with MSIZE parameter
│   ├── original_dataset/
│   │   ├── ffnn/
│   │   └── xgboost/
│   ├── filtered_dataset/
│   │   ├── ffnn/
│   │   └── xgboost/
│   ├── run_test.py
│   └── README.md
├── test_4/                          # Test 4: K-Fold Cross-Validation
│   ├── original_dataset/
│   │   ├── ffnn/
│   │   └── xgboost/
│   ├── filtered_dataset/
│   │   ├── ffnn/
│   │   └── xgboost/
│   ├── run_test.py
│   └── README.md
├── simple_models/                   # Simple baseline models (no USL)
    ├── linear_regression_best.py            # Linear Regression (original)
    ├── linear_regression_best_filtered.py   # Linear Regression (filtered)
    ├── polynomial_regression_best.py        # Polynomial Regression (original)
    ├── polynomial_regression_best_filtered.py
    ├── random_forest_best.py                # Random Forest (original)
    ├── random_forest_best_filtered.py       # Random Forest (filtered)
    ├── ffnn_best_original.py                # FFNN (original)
    ├── ffnn_best_filtered.py                # FFNN (filtered)
    ├── xgboost_best_original.py             # XGBoost (original)
    ├── xgboost_best_filtered.py             # XGBoost (filtered)
    └── artifacts/                           # Pre-trained model weights and artifacts
├── analysis_results/                # Statistical analysis outputs
│   ├── original/                    # Analysis for original dataset
│   │   ├── correlation_matrix_original.csv
│   │   ├── *_heatmap_original.png
│   │   ├── bias_analysis_*.png
│   │   ├── pca_*.png
│   │   └── scatterplots_*/
│   └── filtered/                    # Analysis for filtered dataset
│       └── [same structure as original]
├── comparison_results/              # Model comparison outputs
│   ├── simple_models_original/      # Comparison plots for original dataset
│   ├── simple_models_filtered/      # Comparison plots for filtered dataset
│   └── model_comparison_summary.csv
├── example_usage.py                 # Demonstration script for all tests
├── correlation_analysis.py          # Statistical analysis tool
├── comparison_analysis.py           # Test models comparison tool
├── comparison_simple_models.py      # Simple models comparison tool
├── filter_dataset.py                # Dataset filtering utility
├── model_artifacts.py               # Artifact management utilities
├── original_dataset.csv             # Full dataset (538 records)
├── filtered_dataset.csv             # Filtered dataset (511 records)
├── requirements.txt                 # Python dependencies
├── QUICKSTART.md                    # Quick start guide
├── DATASETS.md                      # Dataset documentation
└── FILE_INVENTORY.md                # Complete file listing
```

## Test Scenarios

### Simple Models (Baseline)
**Directory:** `simple_models/`

Traditional machine learning models without USL integration, serving as baseline comparisons:
- **Linear Regression:** Basic linear relationship modeling
- **Polynomial Regression:** Non-linear relationships with polynomial features
- **Random Forest:** Ensemble decision tree method
- **FFNN:** Feed-Forward Neural Network (simple architecture)
- **XGBoost:** Gradient boosting framework

**Available for both:**
- **Original Dataset:** Full dataset with all measurements
- **Filtered Dataset:** Homogeneous dataset for consistency

These models predict execution time directly from input features (NUM_NODES, MSIZE, BSIZE) without incorporating the Universal Scalability Law.

### Test 1: Input Size-Based Prediction
**Directory:** `test_1/`

Models that use the computed input size as a feature:
- **Inputs:** Number of nodes, INPUT_SIZE (calculated as: 8 × BSIZE² × MSIZE²)
- **USL Calculation:** Based on INPUT_SIZE
- **Models:** FFNN, XGBoost

### Test 2: Matrix Parameters-Based Prediction
**Directory:** `test_2/`

Models that use raw matrix parameters as features:
- **Inputs:** Number of nodes, MSIZE (matrix size), BSIZE (block size)
- **USL Calculation:** Based on INPUT_SIZE
- **Models:** FFNN, XGBoost

### Test 3: USL with MSIZE
**Directory:** `test_3/`

Models using MSIZE for USL calculations:
- **Inputs:** Number of nodes, MSIZE, BSIZE
- **USL Calculation:** Based on MSIZE
- **Models:** FFNN, XGBoost

### Test 4: K-Fold Cross-Validation
**Directory:** `test_4/`

Models trained with K-Fold Cross-Validation (5 folds) for robust performance:
- **Inputs:** Number of nodes, MSIZE, BSIZE
- **USL Calculation:** Based on MSIZE
- **Validation:** 5-fold cross-validation for reliable performance estimates
- **Models:** FFNN, XGBoost
- **Benefits:** More robust hyperparameter selection and reduced overfitting

## Quick Start

### Demo Script

The easiest way to see all models in action:

```bash
python3 example_usage.py
```

This script demonstrates:
- All 4 test approaches with both FFNN and XGBoost
- Batch predictions on multiple configurations
- Ensemble methods combining predictions
- Simple models comparison

### Simple Models

Each simple model can be imported and used directly:

```python
# Import simple models
from simple_models.linear_regression_best import exec_prediction as lr_predict
from simple_models.random_forest_best import exec_prediction as rf_predict
from simple_models.ffnn_best_original import exec_prediction as ffnn_predict
from simple_models.xgboost_best_original import exec_prediction as xgb_predict

# Predict execution time
n_nodes = 16
msize = 8192
bsize = 512

# Get predictions from different models
lr_time = lr_predict(msize, bsize, n_nodes)
rf_time = rf_predict(msize, bsize, n_nodes)
ffnn_time = ffnn_predict(msize, bsize, n_nodes)
xgb_time = xgb_predict(msize, bsize, n_nodes)

print(f"Linear Regression: {lr_time:.2f}s")
print(f"Random Forest: {rf_time:.2f}s")
print(f"FFNN: {ffnn_time:.2f}s")
print(f"XGBoost: {xgb_time:.2f}s")
```

### Complex Models (USL-based)

Each test directory contains subdirectories for both dataset types (original and filtered). Within each, you'll find:

- **ffnn/**: Feed-Forward Neural Network implementation
  - `ffnn_model.py`: Model implementation and prediction function
  - `ffnn_analysis.py`: Analysis and evaluation scripts
  - Model files: `.pth`, `.pkl`, `.json`

- **xgboost/**: XGBoost implementation
  - `xgboost_model.py`: Model implementation and prediction function
  - `xgboost_analysis.py`: Analysis and evaluation scripts
  - Model files: `.json`, `.pkl`

### Running Predictions

**Simple Models (Direct Prediction):**
```python
from simple_models.linear_regression_best import exec_prediction
from simple_models.random_forest_best_filtered import exec_prediction as rf_predict

# Simple prediction without USL
# Function signature: exec_prediction(msize, bsize, num_nodes)
exec_time = exec_prediction(msize=8192, bsize=512, num_nodes=16)
rf_time = rf_predict(8192, 512, 16)  # Can also use positional arguments
```

**Complex Models (USL-based Prediction):**

Each model directory contains a `run_ffnn()` or `run_xgboost()` function that can be used directly:

**For Test 1 (Input Size models):**
```python
from test_1.original_dataset.ffnn.ffnn_model import run_ffnn
from test_1.original_dataset.xgboost.xgboost_model import run_xgboost

# Predict execution time
n_nodes = 16
msize = 8192
bsize = 512

ffnn_time = run_ffnn(n_nodes, msize, bsize)
xgb_time = run_xgboost(n_nodes, msize, bsize)
```

**For Test 2, 3, 4 (Matrix Parameters models):**
```python
from test_2.original_dataset.ffnn.ffnn_model import run_ffnn
from test_2.original_dataset.xgboost.xgboost_model import run_xgboost

# Predict execution time
n_nodes = 16
msize = 8192
bsize = 512

ffnn_time = run_ffnn(n_nodes, msize, bsize)
xgb_time = run_xgboost(n_nodes, msize, bsize)
```

### Model Comparison

To compare simple models vs USL-based complex models:

```python
# Simple models (baseline)
from simple_models.random_forest_best import exec_prediction as rf_predict
from simple_models.xgboost_best_original import exec_prediction as xgb_simple

# Complex models (USL-based)
from test_2.original_dataset.xgboost.xgboost_model import run_xgboost as xgb_complex

msize, bsize, n_nodes = 8192, 512, 16

print(f"Simple XGBoost: {xgb_simple(msize, bsize, n_nodes):.2f}s")
print(f"USL-based XGBoost: {xgb_complex(n_nodes, msize, bsize):.2f}s")
```

### Running Analysis

To run the analysis scripts for model evaluation:

```bash
# Complex models analysis
cd test_1/original_dataset/ffnn/
python ffnn_analysis.py
```

## Model Types

### Simple Models (Baseline)
**Location:** `simple_models/`

Traditional machine learning approaches that predict execution time directly:

1. **Linear Regression**
   - Simple linear relationship between features and execution time
   - Fast training and prediction
   - Good baseline for comparison

2. **Polynomial Regression**
   - Captures non-linear relationships
   - Uses polynomial feature expansion
   - Better fit for complex patterns

3. **Random Forest**
   - Ensemble of decision trees
   - Handles non-linear relationships well
   - Robust to outliers

4. **FFNN (Feed-Forward Neural Network)**
   - Multi-layer perceptron architecture
   - Learns complex non-linear patterns
   - Requires more training data

5. **XGBoost**
   - Gradient boosting framework
   - High accuracy on structured data
   - Feature importance analysis

**Advantages:**
- Simple to implement and understand
- Fast training and prediction
- Good baseline for comparison

**Limitations:**
- No incorporation of parallel computing theory (USL)
- May not capture scalability patterns as effectively

### Complex Models (USL-based)
**Location:** `test_1/`, `test_2/`, `test_3/`, `test_4/`

Advanced models incorporating the Universal Scalability Law:

1. **FFNN (Feed-Forward Neural Network) with USL**
   - Predicts residuals between actual execution time and USL baseline
   - Multi-layer architecture optimized for parallel computing patterns
   - Trained with standardized features for better convergence
   - Uses ensemble methods (K-Fold) in Test 4 for robustness

2. **XGBoost with USL**
   - Gradient boosting optimized for USL residual prediction
   - Captures non-linear scalability patterns
   - Feature importance analysis for understanding key factors
   - Robust to outliers and missing data

**Key Differences from Simple Models:**

| Aspect | Simple Models | USL-based Models |
|--------|--------------|------------------|
| **Approach** | Direct time prediction | Residual prediction (actual - USL) |
| **Scalability Theory** | Not incorporated | USL parameters (α, β) integrated |
| **Input Features** | Raw parameters only | Parameters + USL baseline |
| **Complexity** | Lower (baseline) | Higher (theory-guided) |
| **Accuracy** | Good for general cases | Better for parallel computing scenarios |
| **Interpretability** | High | Moderate (USL + ML) |

**Advantages:**
- Incorporates parallel computing theory (Universal Scalability Law)
- Better captures scalability patterns across different node counts
- Residual learning focuses on hard-to-predict deviations
- More accurate for parallel computing workload predictions

**Test Variations:**
- **Test 1:** Uses INPUT_SIZE as primary feature
- **Test 2:** Uses raw MSIZE and BSIZE parameters
- **Test 3:** USL calculated based on MSIZE parameter
- **Test 4:** Adds K-Fold cross-validation for robust hyperparameter tuning

## Requirements

### For Simple Models
- Python 3.8+
- NumPy
- scikit-learn
- joblib
- (PyTorch for FFNN models)
- (XGBoost for XGBoost models)

### For Complex Models (USL-based)
- Python 3.8+
- PyTorch
- XGBoost
- NumPy
- scikit-learn
- joblib

Install all dependencies:
```bash
pip install torch xgboost numpy scikit-learn joblib matplotlib scipy pandas
```

## Statistical Analysis

### Comprehensive Correlation and Statistical Analysis

The repository includes a comprehensive statistical analysis tool that performs:
- **Bias Analysis**: Input size distribution across different node counts
- **Pearson Correlation**: Linear correlation tests with p-values
- **Spearman Correlation**: Rank correlation tests for non-linear relationships
- **Principal Component Analysis (PCA)**: Dimensionality reduction and feature importance
- **Distribution Analysis**: Histograms, box plots, and outlier detection

```bash
# Analyze both datasets (default)
python3 correlation_analysis.py

# Analyze only original dataset
python3 correlation_analysis.py --dataset original

# Analyze only filtered dataset
python3 correlation_analysis.py --dataset filtered
```

**Generated outputs** (saved in `analysis_results/`):

*Correlation Analysis:*
- `correlation_matrix_*.csv` - Basic correlation matrices
- `correlation_heatmap_*.png` - Visual correlation heatmaps
- `pearson_correlation_heatmap_*.png` - Pearson correlation with values
- `spearman_correlation_heatmap_*.png` - Spearman rank correlation
- `scatterplots_pearson/` - Individual scatterplots for significant Pearson correlations
- `scatterplots_spearman/` - Individual scatterplots for significant Spearman correlations

*Bias Analysis:*
- `bias_analysis_boxplot_*.png` - Input size distribution by node count
- `bias_analysis_heatmap_*.png` - Categorized input size distribution

*PCA Analysis:*
- `pca_scatterplot_*.png` - PC1 vs PC2 scatter plot
- `pca_biplot_*.png` - Biplot with feature vectors overlaid
- `pca_variance_*.png` - Explained variance by component
- `pca_loadings_*.csv` - Component loadings for each feature

*Distribution Analysis:*
- `exec_time_distribution_*.png` - EXEC_TIME histogram
- `all_distributions_*.png` - All feature distributions
- `box_plots_*.png` - Box plots for outlier detection
- `summary_statistics_*.txt` - Complete statistical summary

**Key insights from analysis:**
- Feature correlations with execution time (Pearson and Spearman)
- Distribution characteristics (skewness, kurtosis)
- Outlier detection using IQR method
- Bias assessment across different configurations
- Principal components explaining data variance
- Statistical significance testing (p-values < 0.05)

### Model Comparison Analysis

**Available Scripts:**

1. **`comparison_analysis.py`** - Compare all test models (test_1 through test_4)
2. **`comparison_simple_models.py`** - Compare all simple baseline models

```bash
# Compare all test models (test_1 through test_4)
python3 comparison_analysis.py

# Compare simple models for specific dataset
python3 comparison_simple_models.py --dataset original
python3 comparison_simple_models.py --dataset filtered
python3 comparison_simple_models.py --dataset both  # Compare both datasets
```

**Generated outputs:**

*Test Models Comparison* (from `comparison_analysis.py`):
- `all_tests_predictions.csv` - All predictions from test_1 through test_4
- `comparison_results/*.png` - Accuracy/safety plots for each test and model type

*Simple Models Comparison* (from `comparison_simple_models.py`):
- `comparison_simple_models_original.csv` - Predictions on original dataset
- `comparison_simple_models_filtered.csv` - Predictions on filtered dataset
- `comparison_simple_models_summary.csv` - Performance summary with metrics
- `comparison_results/simple_models_*/*.png` - Individual model accuracy plots

**Safety Classification:**
- **Correct:** Prediction error ≤ 10%
- **Safe (overestimation):** Prediction > 10% higher than actual (conservative)
- **Unsafe (underestimation):** Prediction > 10% lower than actual (risky)

Each plot includes:
- Scatter plot of predicted vs actual execution times
- Color-coded safety classification
- Ideal prediction line (y=x)
- Statistics showing percentage of correct, safe, and unsafe predictions

**Key insights:**
- Model accuracy across different test configurations
- Safety profile of each model (conservative vs risky predictions)
- Comparison between FFNN and XGBoost approaches
- Original vs filtered dataset performance

## Model Details

### Universal Scalability Law (USL)
All models incorporate the Universal Scalability Law to capture parallel computing performance characteristics:

```
USL(n) = n / (1 + α(n-1) + βn(n-1))
```

Where:
- `n`: number of workers (nodes - 1)
- `α`: contention parameter
- `β`: coherency parameter

### Residual Prediction Approach
Models predict the residual (difference) between the actual execution time and the USL prediction, improving accuracy over direct time prediction.

## Dataset Information

- **Original Dataset:** Full dataset with all available measurements (538 records)
- **Filtered Dataset:** Homogeneous dataset filtered for consistency across different node counts (511 records)

The filtered dataset generally provides more stable predictions by removing outliers and ensuring consistent data distribution.

### Dataset Filtering

To regenerate the filtered dataset:

```bash
python3 filter_dataset.py
```

This script filters the original dataset by removing entries with INPUT_SIZE below a certain threshold to ensure more homogeneous data distribution across different node configurations.

