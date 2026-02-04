# File Inventory - PUBLIC Folder

## Overview
This document provides a complete inventory of all files and their purposes in the PUBLIC folder.

## Root Directory Files

### Datasets
- **`original_dataset.csv`** - Complete unfiltered dataset (538 samples)
- **`filtered_dataset.csv`** - Filtered dataset for homogeneous analysis (currently identical to original)

### Scripts
- **`correlation_analysis.py`** - Statistical analysis tool for correlation matrices and heatmap generation
- **`filter_dataset.py`** - Script to filter dataset by INPUT_SIZE threshold
- **`example_usage.py`** - Comprehensive demonstration of all 4 tests

### Documentation
- **`README.md`** - Main documentation with project overview and usage instructions
- **`QUICKSTART.md`** - Quick start guide for immediate usage
- **`DATASETS.md`** - Dataset documentation and schema description
- **`SUMMARY.md`** - Project summary and model comparison
- **`STRUCTURE.txt`** - Directory structure reference
- **`requirements.txt`** - Python package dependencies

## Analysis Results Directory

**`analysis_results/`** - Contains statistical analysis outputs

### Original Dataset Analysis
- `original/correlation_matrix_original.csv` - Correlation coefficients matrix
- `original/correlation_heatmap_original.png` - Visual correlation heatmap
- `original/exec_time_distribution_original.png` - EXEC_TIME histogram
- `original/all_distributions_original.png` - All feature distributions
- `original/box_plots_original.png` - Box plots for outlier detection
- `original/summary_statistics_original.txt` - Statistical summary text file

### Filtered Dataset Analysis
- `filtered/correlation_matrix_filtered.csv` - Correlation coefficients matrix
- `filtered/correlation_heatmap_filtered.png` - Visual correlation heatmap
- `filtered/exec_time_distribution_filtered.png` - EXEC_TIME histogram
- `filtered/all_distributions_filtered.png` - All feature distributions
- `filtered/box_plots_filtered.png` - Box plots for outlier detection
- `filtered/summary_statistics_filtered.txt` - Statistical summary text file

### Documentation
- `analysis_results/README.md` - Explanation of analysis results and methodology

## Test Directories

Each test directory follows the same structure:

### Test 1: Input Size-Based Prediction (`test_1/`)

#### Root Files
- `run_test.py` - Unified interface for all models
- `README.md` - Test-specific documentation

#### Original Dataset - FFNN (`original_dataset/ffnn/`)
- `ffnn_model.py` - FFNN prediction interface
- `ffnn_analysis.py` - Model evaluation and metrics
- `model_ffnn.pth` - Trained PyTorch model
- `scaler_X_ffnn.pkl` - Input feature scaler
- `scaler_y_ffnn.pkl` - Output target scaler
- `usl_parameters_ffnn.json` - USL model parameters
- `usl_residuals_ffnn.csv` - USL residuals data

#### Original Dataset - XGBoost (`original_dataset/xgboost/`)
- `xgboost_model.py` - XGBoost prediction interface
- `xgboost_analysis.py` - Model evaluation and metrics
- `model_xgboost-model.json` - Trained XGBoost model
- `scaler_X_xgboost-model.pkl` - Input feature scaler
- `scaler_y_xgboost-model.pkl` - Output target scaler
- `usl_parameters_xgboost-model.json` - USL model parameters
- `usl_residuals_xgboost-model.csv` - USL residuals data

#### Filtered Dataset - FFNN (`filtered_dataset/ffnn/`)
- Same structure as original_dataset/ffnn/
- Models trained on filtered dataset

#### Filtered Dataset - XGBoost (`filtered_dataset/xgboost/`)
- Same structure as original_dataset/xgboost/
- Models trained on filtered dataset

### Test 2: Matrix Parameters-Based Prediction (`test_2/`)
- Same file structure as Test 1
- Different input features (MSIZE, BSIZE instead of INPUT_SIZE)
- USL scaling based on INPUT_SIZE

### Test 3: USL with MSIZE Parameter (`test_3/`)
- Same file structure as Test 1
- Different input features (MSIZE, BSIZE)
- USL scaling based on MSIZE (simplified approach)

### Test 4: K-Fold Cross-Validation (`test_4/`)
- Same file structure as Test 1
- Models trained with 5-fold cross-validation
- Enhanced robustness through ensemble approach

## Usage Patterns

### For Users
Most important files for end users:
1. `README.md` - Start here
2. `QUICKSTART.md` - Immediate usage guide
3. `example_usage.py` - Working examples
4. `test_*/run_test.py` - Direct prediction interface
5. `correlation_analysis.py` - Dataset exploration

### For Developers
Most important files for understanding implementation:
1. `test_*/original_dataset/ffnn/ffnn_model.py` - FFNN architecture
2. `test_*/original_dataset/xgboost/xgboost_model.py` - XGBoost interface
3. `test_*/original_dataset/ffnn/ffnn_analysis.py` - Evaluation methodology
4. `usl_parameters_*.json` - Model hyperparameters
5. `correlation_analysis.py` - Statistical methodology

### For Researchers
Most important files for analysis:
1. `analysis_results/` - Statistical insights
2. `DATASETS.md` - Dataset description
3. `SUMMARY.md` - Model comparison results
4. `test_*/original_dataset/*/usl_residuals_*.csv` - Prediction residuals
5. `correlation_analysis.py` - Reproducible analysis

## Maintenance Notes

### Regenerating Files
To regenerate analysis files:
```bash
python3 correlation_analysis.py --dataset both
```

To regenerate filtered dataset:
```bash
python3 filter_dataset.py
```

### Versioning
- All model files are final trained versions
- Scalers match their corresponding models
- USL parameters are optimized for each test case
- Analysis results reflect current dataset state

## Quality Assurance

All files have been:
- ✅ Tested and verified working
- ✅ Documented with inline comments
- ✅ Structured for easy navigation
- ✅ Optimized for publication readiness
- ✅ Cross-referenced in documentation

Last updated: February 3, 2026
