# Test 2: Matrix Parameters-Based Prediction

## Overview

This test uses raw matrix parameters (MSIZE and BSIZE) directly as input features, rather than computing a combined INPUT_SIZE. This approach allows the models to learn the relationships between matrix dimensions and execution time independently.

## Model Approach

### Input Features
- **Number of nodes**: Used as-is in the model
- **MSIZE**: Matrix size (number of rows/columns)
- **BSIZE**: Block size

### USL Calculation
The Universal Scalability Law calculation uses the INPUT_SIZE parameter (calculated internally):
```
INPUT_SIZE = 8 × BSIZE² × MSIZE²
USL(n) = (baseline_time × INPUT_SIZE) / (n / (1 + α(n-1) + βn(n-1)))
```

### Prediction Method
Models predict the residual (difference) between actual execution time and USL prediction:
```
Predicted Time = USL Time + Residual Prediction
```

## Directory Structure

```
test-3-input-msize-bsize/
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

## Advantages of This Approach

1. **Independent Feature Learning**: Models can learn separate relationships for MSIZE and BSIZE
2. **Non-linear Interactions**: Neural networks can capture complex interactions between matrix parameters
3. **Flexibility**: Can adapt to different scaling behaviors of rows vs blocks
4. **Better Generalization**: May generalize better to unseen matrix configurations

## Dataset Comparison

### Original Dataset
- Contains all available measurements
- Full range of matrix configurations
- Captures diverse execution patterns

### Filtered Dataset
- Homogeneous dataset filtered for consistency
- Outliers removed for improved stability
- More uniform distribution across node counts

## Model Details

### FFNN Architecture
- **Input Layer**: 3 features (n_workers, msize, bsize)
- **Hidden Layers**: [64, 64] neurons with ReLU activation
- **Output Layer**: 1 neuron (residual prediction)
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error

### XGBoost Configuration
- **Objective**: reg:squarederror
- **Booster**: gbtree
- **Hyperparameters**: Optimized via cross-validation
- **Feature Importance**: Can analyze importance of MSIZE vs BSIZE

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
- Feature importance (for XGBoost)

Check the `BEST_MODEL.json` file in each model directory for detailed performance metrics.

## Notes

- n_workers is calculated as (n_nodes - 1) internally
- INPUT_SIZE is computed inside the function for USL calculation
- USL parameters (α, β, baseline_time) are pre-computed and stored in JSON files
- All scalers are pre-fitted and saved for consistent preprocessing
- The 3-feature approach allows models to learn non-standard scaling relationships
