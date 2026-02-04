# Test 4: K-Fold Cross-Validation

This test demonstrates the use of K-Fold Cross-Validation (5 folds) for training and evaluating the models, providing more robust performance estimates.

## Description

K-Fold Cross-Validation is a resampling procedure used to evaluate machine learning models on limited data samples. The dataset is divided into K equal folds (5 in this case), and the model is trained K times, each time using K-1 folds for training and the remaining fold for validation.

## Input Features

The models use the following inputs:
- **Number of nodes:** The number of compute nodes (used as is)
- **MSIZE:** Matrix size (number of rows/columns)
- **BSIZE:** Block size of the matrix

## USL Calculation

The Universal Scalability Law parameters (α and β) are calculated using **MSIZE** as the scaling parameter.

## Validation Strategy

**K-Fold Cross-Validation with 5 folds:**
- Dataset is split into 5 equal parts
- Each fold serves as validation set once while the other 4 are used for training
- Final model is trained on the entire dataset using the best hyperparameters
- Provides more reliable performance estimates than single train-test split

## Datasets

### Original Dataset
Full dataset with all available measurements, providing comprehensive training data.

### Filtered Dataset
Homogeneous dataset filtered for consistency across different node counts, reducing noise and improving model stability.

## Models

### FFNN (Feed-Forward Neural Network)
- Architecture optimized through cross-validation
- Trained with early stopping based on validation performance
- Predicts residuals from USL predictions

### XGBoost (Extreme Gradient Boosting)
- Hyperparameters tuned using cross-validation
- Robust to overfitting through regularization
- Predicts residuals from USL predictions

## Usage

### Running Models

```python
from test_5.original_dataset.ffnn.ffnn_model import run_ffnn
from test_5.original_dataset.xgboost.xgboost_model import run_xgboost

# Predict execution time
n_nodes = 2
msize = 9
bsize = 500

ffnn_time = run_ffnn(n_nodes, msize, bsize)
xgb_time = run_xgboost(n_nodes, msize, bsize)

print(f"FFNN prediction: {ffnn_time:.2f} seconds")
print(f"XGBoost prediction: {xgb_time:.2f} seconds")
```

### Using the Unified Interface

```bash
cd test_5
python3 run_test.py
```

### Running Analysis

```bash
cd original-dataset/ffnn
python3 ffnn_analysis.py

cd ../xgboost
python3 xgboost_analysis.py
```

## Model Files

Each model directory contains:
- `{model}_model.py`: Model loading and prediction functions
- `{model}_analysis.py`: Analysis and evaluation scripts
- Model weights: `.pth` (FFNN) or `.json` (XGBoost)
- Scalers: `.pkl` files for feature and target normalization
- `usl_parameters_{model}.json`: USL parameters (α, β, baseline_time)
- `usl_residuals_{model}.csv`: Training residuals for analysis
- `BEST_MODEL.json`: Model configuration and performance metrics
- `output_analysis.txt`: Detailed analysis results

## Performance Metrics

K-Fold Cross-Validation provides:
- Mean performance across folds
- Standard deviation of performance
- Confidence intervals
- More reliable comparison between models

## Requirements

```bash
pip install torch xgboost numpy scikit-learn pandas joblib matplotlib
```

## References

- K-Fold Cross-Validation: Scikit-learn documentation
- Universal Scalability Law: Gunther, N.J. (2007)
