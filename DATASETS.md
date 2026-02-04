# Dataset Documentation

Here, I explain the structure of the datasets used to train and evaluate all models.

## Dataset Files

### `original_dataset.csv`
The complete, unfiltered dataset containing all measurements from the COMPSs matrix multiplication application.

- **Size:** 539 entries
- **Description:** Full dataset with all available measurements
- **Use case:** Training models that need comprehensive data coverage

### `filtered_dataset.csv`
A filtered version of the dataset designed to be more homogeneous across different node counts.

- **Size:** 511 entries (filtered for consistency)
- **Description:** Dataset filtered to ensure consistent measurements across different configurations
- **Use case:** Training models with more stable and predictable data distributions

## Dataset Schema

Both datasets contain the following columns:

| Column | Description |
|--------|-------------|
| `EXEC_TIME` | Execution time in seconds (target variable) |
| `MSIZE` | Matrix size (number of rows/columns) |
| `BSIZE` | Block size of the matrix |
| `NUM_NODES` | Number of compute nodes used |
| `NUM_CPUS` | Total number of CPUs used |
| `CPUS_PER_NODE` | Number of CPUs per node |
| `CPU_AVG` | Average CPU usage percentage |
| `MEM_AVG` | Average memory usage percentage |
| `CPU_AVG_USED` | Average CPU cores used |
| `MEM_AVG_USED` | Average memory used (GB) |

## Dataset Usage in Tests

Each test scenario uses both datasets to compare model performance:

### Test 1: Input Size-Based Prediction
- **Original dataset:** Models trained on `original_dataset.csv`
- **Filtered dataset:** Models trained on `filtered_dataset.csv`
- **Purpose:** Compare how dataset quality affects prediction accuracy

### Test 2: Matrix Parameters-Based Prediction
- Uses both datasets with raw MSIZE and BSIZE as features
- **Purpose:** Evaluate direct parameter-based modeling

### Test 3: USL with MSIZE
- Uses both datasets with MSIZE for USL calculation
- **Purpose:** Improve the model with theoretical baseline using USL scaling approach

### Test 4: K-Fold Cross-Validation
- Uses both datasets with 5-fold cross-validation
- **Purpose:** Robust performance estimation and reduced overfitting

## Filtering Methodology

The filtered dataset was created to:
1. Remove outliers and anomalous measurements
2. Ensure consistent data distribution across node counts
3. Improve model stability and generalization
4. Reduce variance in predictions

### Benefits of Filtered Dataset:
- ✓ More consistent predictions across different configurations
- ✓ Reduced impact of measurement noise
- ✓ Better model generalization
- ✓ More reliable performance metrics

## Data Preprocessing

All models apply the following preprocessing steps:
1. **Feature Scaling:** StandardScaler for input features
2. **Target Scaling:** StandardScaler for execution times
3. **USL Calculation:** Universal Scalability Law predictions
4. **Residual Prediction:** Models predict the difference between actual and USL-predicted times

## Additional Information

For more details about:
- **Model implementations:** See individual test directories
- **Training procedures:** See `{model}_model.py` files in each test
- **Analysis results:** See `{model}_analysis.py` and `output_analysis.txt` files
- **Performance metrics:** See `BEST_MODEL.json` files in each model directory
