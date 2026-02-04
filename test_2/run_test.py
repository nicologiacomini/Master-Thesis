"""
Test 2: Matrix Parameters-Based Prediction
Unified interface for running both FFNN and XGBoost models on both datasets
"""

import sys
import os


def predict_execution_time(n_nodes, msize, bsize, model='both', dataset='both'):
    """
    Predict COMPSs execution time using ML models
    
    Parameters:
    -----------
    n_nodes : int
        Number of compute nodes
    msize : int
        Matrix size (number of rows/columns)
    bsize : int
        Block size
    model : str, optional (default='both')
        Model to use: 'ffnn', 'xgboost', or 'both'
    dataset : str, optional (default='both')
        Dataset to use: 'original', 'filtered', or 'both'
    
    Returns:
    --------
    dict : Dictionary with predictions for each model/dataset combination
    """
    results = {}
    
    # Original dataset predictions
    if dataset in ['original', 'both']:
        if model in ['ffnn', 'both']:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'original_dataset', 'ffnn'))
            import ffnn_model
            results['ffnn_original'] = ffnn_model.run_ffnn(n_nodes, msize, bsize)
            sys.path.pop(0)
        
        if model in ['xgboost', 'both']:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'original_dataset', 'xgboost'))
            import xgboost_model
            results['xgboost_original'] = xgboost_model.run_xgboost(n_nodes, msize, bsize)
            sys.path.pop(0)
    
    # Filtered dataset predictions
    if dataset in ['filtered', 'both']:
        if model in ['ffnn', 'both']:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'filtered_dataset', 'ffnn'))
            import ffnn_model as ffnn_model_filtered
            results['ffnn_filtered'] = ffnn_model_filtered.run_ffnn(n_nodes, msize, bsize)
            sys.path.pop(0)
        
        if model in ['xgboost', 'both']:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'filtered_dataset', 'xgboost'))
            import xgboost_model as xgboost_model_filtered
            results['xgboost_filtered'] = xgboost_model_filtered.run_xgboost(n_nodes, msize, bsize)
            sys.path.pop(0)
    
    return results


def main():
    """Example usage"""
    # Example parameters
    n_nodes = 2
    msize = 9
    bsize = 500
    
    print("=" * 70)
    print("Test 2: Matrix Parameters-Based Prediction")
    print("=" * 70)
    print(f"\nInput parameters:")
    print(f"  - Number of nodes: {n_nodes}")
    print(f"  - Matrix size (MSIZE): {msize}")
    print(f"  - Block size (BSIZE): {bsize}")
    
    # Calculate input size for reference
    input_size = 8 * (bsize ** 2) * (msize ** 2)
    print(f"  - Computed INPUT_SIZE: {input_size:,} bytes")
    
    # Get predictions
    print("\n" + "-" * 70)
    print("Predictions:")
    print("-" * 70)
    
    predictions = predict_execution_time(n_nodes, msize, bsize)
    
    for model_name, pred_time in predictions.items():
        print(f"  {model_name:20s}: {pred_time:>12.2f} seconds")
    
    # Calculate average and variance
    pred_values = list(predictions.values())
    avg_time = sum(pred_values) / len(pred_values)
    variance = sum((x - avg_time) ** 2 for x in pred_values) / len(pred_values)
    std_dev = variance ** 0.5
    
    print("\n" + "-" * 70)
    print(f"  {'Average':20s}: {avg_time:>12.2f} seconds")
    print(f"  {'Std. Deviation':20s}: {std_dev:>12.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
