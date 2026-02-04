# Quick Start Guide

## Installation

1. **Install dependencies:**
```bash
pip install torch xgboost numpy scikit-learn joblib matplotlib scipy pandas
```

2. **Navigate to a test directory:**
```bash
cd test-1-input-inputsize/
```

## Dataset Analysis (Optional)

Before running predictions, you can analyze the datasets to understand feature correlations and distributions:

### Statistical Analysis

```bash
# Run correlation analysis on both datasets
python3 correlation_analysis.py

# Or analyze specific dataset
python3 correlation_analysis.py --dataset original
python3 correlation_analysis.py --dataset filtered
```

This generates:
- Correlation heatmaps (Pearson and Spearman)
- Feature distribution plots
- Bias analysis
- PCA analysis
- Statistical summaries
- Outlier analysis

### Model Comparison Analysis

```bash
# Compare accuracy and safety of all models
python3 model_comparison_analysis.py

# Or analyze specific dataset
python3 model_comparison_analysis.py --dataset original
python3 model_comparison_analysis.py --dataset filtered
```

This generates:
- Prediction CSV files with all model results
- Accuracy and safety scatter plots for each model
- Performance statistics (correct, safe, unsafe predictions)

Results are saved in `analysis_results/original/` and `analysis_results/filtered/` directories.

## Running Models

### Option 1: Use the Unified Interface (Recommended)

```bash
python run_test.py
```

This runs all models (FFNN and XGBoost) on both datasets and displays predictions.

### Option 2: Python API

```python
from test_1.run_test import predict_execution_time

# Get predictions
predictions = predict_execution_time(
    n_nodes=2,      # Number of compute nodes
    msize=9,      # Matrix size
    bsize=500,       # Block size
    model='both',    # 'ffnn', 'xgboost', or 'both'
    dataset='both'   # 'original', 'filtered', or 'both'
)

print(predictions)
```

### Option 3: Direct Model Import

```python
from test_1.original_dataset.ffnn.ffnn_model import run_ffnn
from test_1.filtered_dataset.xgboost.xgboost_model import run_xgboost

# Single model prediction
time_ffnn = run_ffnn(n_nodes=2, msize=9, bsize=500)
time_xgb = run_xgboost(n_nodes=2, msize=9, bsize=500)

print(f"FFNN prediction: {time_ffnn:.2f} seconds")
print(f"XGBoost prediction: {time_xgb:.2f} seconds")
```

## Choosing a Test

| Test | Use Case | Inputs | USL Scaling |
|------|----------|--------|-------------|
| **Test 1** | Standard input size prediction | n_nodes, INPUT_SIZE | INPUT_SIZE |
| **Test 2** | Compare dataset filtering impact | n_nodes, MSIZE, BSIZE | INPUT_SIZE |
| **Test 3** | Matrix parameter-based prediction | n_nodes, MSIZE, BSIZE | MSIZE |
| **Test 4** | MSIZE-based USL scaling | n_nodes, MSIZE, BSIZE | MSIZE |

## Troubleshooting

### Import Errors
If you get import errors, ensure you're running from the PUBLIC directory or add it to your Python path:
```python
import sys
sys.path.insert(0, '/path/to/PUBLIC')
```

### Missing Dependencies
```bash
pip install torch xgboost numpy scikit-learn joblib
```

### Model Loading Errors
Ensure all `.pth`, `.json`, and `.pkl` files are in the same directory as the model Python files.

## Next Steps

- Read the README in each test directory for detailed information
- Run the analysis scripts to see model performance metrics
- Experiment with different input parameters
- Compare predictions across different tests and datasets
