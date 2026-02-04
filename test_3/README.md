# Test 3: USL with MSIZE Parameter

## Overview

This test explores an alternative USL (Universal Scalability Law) calculation approach where MSIZE is used directly instead of the computed INPUT_SIZE. This can be more appropriate when the dominant scaling factor is the matrix dimension rather than the total data size.

## Model Approach

### Input Features
- **Number of nodes**: Used as-is in the model
- **MSIZE**: Matrix size (number of rows/columns)
- **BSIZE**: Block size

### USL Calculation (Key Difference)
Unlike other tests, this approach uses **MSIZE** for USL scaling:
```
USL(n) = (baseline_time × MSIZE) / (n / (1 + α(n-1) + βn(n-1)))
```

This differs from Test 3, which uses INPUT_SIZE in the USL calculation.

### Prediction Method
Models predict the residual (difference) between actual execution time and USL prediction:
```
Predicted Time = USL Time + Residual Prediction
```

## Directory Structure

```
test-4-usl-msize/
├── README.md (this file)
├── run_test.py (unified interface)
├── original-dataset/
│   ├── ffnn/
│   │   ├── ffnn_model.py
│   │   ├── ffnn_analysis.py
│   │   ├── model_ffnn.pth
│   │   ├── scaler_X_ffnn.pkl
│   │   ├── scaler_y_ffnn.pkl
│   │   ├── usl_parameters_ffnn.json
│   │   └── BEST_MODEL.json
│   └── xgboost/
│       ├── xgboost_model.py
│       ├── xgboost_analysis.py
│       ├── model_xgboost-model.json
│       ├── scaler_X_xgboost-model.pkl
│       ├── scaler_y_xgboost-model.pkl
│       ├── usl_parameters_xgboost-model.json
│       └── BEST_MODEL.json
└── filtered-dataset/
    ├── ffnn/ (same structure)
    └── xgboost/ (same structure)
```

## Usage

### Quick Start with Unified Interface

```python
from run_test import predict_execution_time

# Get predictions from all models
predictions = predict_execution_time(
    n_nodes=2,
    msize=9,
    bsize=500,
    model='both',      # 'ffnn', 'xgboost', or 'both'
    dataset='both'     # 'original', 'filtered', or 'both'
)

print(predictions)
# Predictions:
# ----------------------------------------------------------------------
#   ffnn_filtered         :     278.06 seconds
#   ffnn_original         :     278.06 seconds
#   xgboost_filtered      :     227.49 seconds
#   xgboost_original      :     227.49 seconds
```

### Run the example script
```bash
python run_test.py
```

### Using Individual Models

**FFNN Model (Original Dataset):**
```python
from original_dataset.ffnn.ffnn_model import run_ffnn

predicted_time = run_ffnn(n_nodes=16, msize=8192, bsize=512)
print(f"Predicted execution time: {predicted_time:.2f} seconds")
```

**XGBoost Model (Filtered Dataset):**
```python
from filtered_dataset.xgboost.xgboost_model import run_xgboost

predicted_time = run_xgboost(n_nodes=16, msize=8192, bsize=512)
print(f"Predicted execution time: {predicted_time:.2f} seconds")
```

## Why MSIZE for USL?

This approach is beneficial when:

1. **Matrix Dimension Dominates**: When the number of rows/columns is the primary computational factor
2. **Block Size is Secondary**: Block size affects data locality but not the fundamental algorithm complexity
3. **Simplicity**: MSIZE provides a simpler scaling metric than combined INPUT_SIZE
4. **Computational Complexity**: Many matrix algorithms scale with O(n³) or O(n²) where n is MSIZE

## Dataset Comparison

### Original Dataset
- Contains all available measurements
- Full range of matrix configurations
- May show different scaling patterns

### Filtered Dataset
- Homogeneous dataset filtered for consistency
- Outliers removed for improved stability
- More predictable scaling behavior

## Model Details

### FFNN Architecture
- **Input Layer**: 3 features (n_workers, msize, bsize)
- **Hidden Layers**: [32, 32] neurons with Sigmoid activation
- **Output Layer**: 1 neuron (residual prediction)
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error

### XGBoost Configuration
- **Objective**: reg:squarederror
- **Booster**: gbtree
- **Hyperparameters**: Optimized via cross-validation

## Running Analysis

To analyze model performance and generate visualizations:

```bash
# FFNN Analysis
cd original-dataset/ffnn/
python ffnn_analysis.py

# XGBoost Analysis
cd original-dataset/xgboost/
python xgboost_analysis.py
```

## Requirements

```bash
pip install torch xgboost numpy scikit-learn joblib matplotlib pandas
```

## Performance Metrics

Both models provide:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Residual analysis plots

Check the `BEST_MODEL.json` file in each model directory for detailed performance metrics.

## Notes

- n_workers is calculated as (n_nodes - 1) internally
- INPUT_SIZE is still computed but only used internally, not for USL
- USL parameters (α, β, baseline_time) are optimized for MSIZE scaling
- All scalers are pre-fitted and saved for consistent preprocessing
- The simpler USL formulation may lead to better generalization in some cases
