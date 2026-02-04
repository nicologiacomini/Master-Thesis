#!/usr/bin/env python3
"""
Example: Using All Test Models
Demonstrates how to use all 4 tests and compare their predictions
"""

import sys
import os
import importlib.util


def import_run_test(test_dir):
    """Dynamically import run_test module from a test directory"""
    module_path = os.path.join(os.path.dirname(__file__), test_dir, 'run_test.py')
    spec = importlib.util.spec_from_file_location("run_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_predictions(predictions, title):
    """Helper function to format prediction results"""
    print(f"\n{title}")
    print("-" * 70)
    for model_name, pred_time in sorted(predictions.items()):
        print(f"  {model_name:22s}: {pred_time:>10.2f} seconds")
    
    values = list(predictions.values())
    avg = sum(values) / len(values)
    std = (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5
    print(f"\n  {'Average':22s}: {avg:>10.2f} seconds")
    print(f"  {'Std. Deviation':22s}: {std:>10.2f} seconds")


def run_all_tests():
    """Run predictions on all 4 tests"""
    
    # Test parameters
    n_nodes = 2
    msize = 9
    bsize = 500
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 80)
    print(f"\nExecution Parameters:")
    print(f"  - Number of nodes: {n_nodes}")
    print(f"  - Matrix size (MSIZE): {msize}")
    print(f"  - Block size (BSIZE): {bsize}")
    
    input_size = 8 * (bsize ** 2) * (msize ** 2)
    print(f"  - Computed INPUT_SIZE: {input_size:,} bytes")
    print(f"  - INPUT_SIZE (GB): {input_size / (1024**3):.2f} GB")
    
    # Test 1: Input Size-Based Prediction
    print("\n" + "=" * 80)
    print("TEST 1: INPUT SIZE-BASED PREDICTION")
    print("=" * 80)
    print("Approach: Uses computed INPUT_SIZE as feature")
    print("USL Scaling: Based on INPUT_SIZE")
    
    test1_module = import_run_test('test_1')
    test1_results = test1_module.predict_execution_time(n_nodes, msize, bsize)
    format_predictions(test1_results, "Predictions:")
    
    # Test 2: Matrix Parameters-Based Prediction
    print("\n" + "=" * 80)
    print("TEST 2: MATRIX PARAMETERS-BASED PREDICTION")
    print("=" * 80)
    print("Approach: Uses MSIZE and BSIZE as separate features")
    print("USL Scaling: Based on INPUT_SIZE")
    
    test2_module = import_run_test('test_2')
    test2_results = test2_module.predict_execution_time(n_nodes, msize, bsize)
    format_predictions(test2_results, "Predictions:")
    
    # Test 3: USL with MSIZE
    print("\n" + "=" * 80)
    print("TEST 3: USL WITH MSIZE PARAMETER")
    print("=" * 80)
    print("Approach: Uses MSIZE and BSIZE as separate features")
    print("USL Scaling: Based on MSIZE (simplified)")
    
    test3_module = import_run_test('test_3')
    test3_results = test3_module.predict_execution_time(n_nodes, msize, bsize)
    format_predictions(test3_results, "Predictions:")
    
    # Test 4: K-Fold Cross-Validation
    print("\n" + "=" * 80)
    print("TEST 4: K-FOLD CROSS-VALIDATION")
    print("=" * 80)
    print("Approach: Uses MSIZE and BSIZE as separate features")
    print("USL Scaling: Based on MSIZE with 5-fold CV")
    
    test4_module = import_run_test('test_4')
    test4_results = test4_module.predict_execution_time(n_nodes, msize, bsize)
    format_predictions(test4_results, "Predictions:")
    
    # Overall Comparison
    print("\n" + "=" * 80)
    print("OVERALL COMPARISON")
    print("=" * 80)
    
    # Collect all filtered dataset predictions for fair comparison
    all_filtered = {
        'Test 1 - FFNN': test1_results.get('ffnn_filtered', 0),
        'Test 1 - XGBoost': test1_results.get('xgboost_filtered', 0),
        'Test 2 - FFNN': test2_results.get('ffnn_filtered', 0),
        'Test 2 - XGBoost': test2_results.get('xgboost_filtered', 0),
        'Test 3 - FFNN': test3_results.get('ffnn_filtered', 0),
        'Test 3 - XGBoost': test3_results.get('xgboost_filtered', 0),
        'Test 4 - FFNN': test4_results.get('ffnn_filtered', 0),
        'Test 4 - XGBoost': test4_results.get('xgboost_filtered', 0),
    }
    
    format_predictions(all_filtered, "All Models (Filtered Dataset):")
    
    # Best and worst predictions
    best_model = min(all_filtered.items(), key=lambda x: x[1])
    worst_model = max(all_filtered.items(), key=lambda x: x[1])
    
    print(f"\nBest (fastest) prediction: {best_model[0]}: {best_model[1]:.2f} seconds")
    print(f"Worst (slowest) prediction: {worst_model[0]}: {worst_model[1]:.2f} seconds")
    print(f"Prediction range: {worst_model[1] - best_model[1]:.2f} seconds")
    
    # Ensemble prediction
    ensemble_avg = sum(all_filtered.values()) / len(all_filtered)
    print(f"\nEnsemble prediction (average of all): {ensemble_avg:.2f} seconds")
    
    print("\n" + "=" * 80)


def demonstrate_specific_use_case():
    """Demonstrate a specific use case"""
    
    print("\n\n" + "=" * 80)
    print("SPECIFIC USE CASE: PRODUCTION DEPLOYMENT")
    print("=" * 80)
    print("\nScenario: You want the most reliable prediction for production scheduling")
    print("Recommendation: Use filtered dataset with ensemble of both models\n")
    
    # Import the most reliable models
    from test_1.filtered_dataset.ffnn.ffnn_model import run_ffnn
    from test_1.filtered_dataset.xgboost.xgboost_model import run_xgboost
    
    # Parameters
    n_nodes = 2
    msize = 9
    bsize = 500
    
    print(f"Parameters: {n_nodes} nodes, MSIZE={msize}, BSIZE={bsize}")
    
    # Get predictions
    ffnn_time = run_ffnn(n_nodes, msize, bsize)
    xgb_time = run_xgboost(n_nodes, msize, bsize)
    
    # Ensemble
    ensemble = (ffnn_time + xgb_time) / 2
    uncertainty = abs(ffnn_time - xgb_time) / 2
    
    print(f"\nResults:")
    print(f"  FFNN prediction:      {ffnn_time:>10.2f} seconds")
    print(f"  XGBoost prediction:   {xgb_time:>10.2f} seconds")
    print(f"  Ensemble prediction:  {ensemble:>10.2f} ± {uncertainty:.2f} seconds")
    print(f"\nUse {ensemble:.0f} seconds for scheduling with {uncertainty:.0f}s buffer")
    
    print("=" * 80)


def demonstrate_batch_predictions():
    """Demonstrate batch predictions for multiple configurations"""
    
    print("\n\n" + "=" * 80)
    print("BATCH PREDICTIONS: SCALING ANALYSIS")
    print("=" * 80)
    
    from test_1.filtered_dataset.xgboost.xgboost_model import run_xgboost
    
    # Different configurations
    configs = [
        {'n_nodes': 2, 'msize': 9, 'bsize': 500},
        {'n_nodes': 5, 'msize': 43, 'bsize': 345},
        {'n_nodes': 6, 'msize': 37, 'bsize': 200},
    ]
    
    print("\nPredicting execution time for different scales:\n")
    print(f"{'Nodes':>6} {'MSIZE':>8} {'BSIZE':>6} {'Predicted Time':>15} {'Speedup':>10}")
    print("-" * 60)
    
    baseline_time = None
    for config in configs:
        pred_time = run_xgboost(**config)
        
        if baseline_time is None:
            baseline_time = pred_time
            speedup = 1.0
        else:
            speedup = baseline_time / pred_time
        
        print(f"{config['n_nodes']:>6} {config['msize']:>8} {config['bsize']:>6} "
              f"{pred_time:>12.2f} sec {speedup:>9.2f}x")
    
    print("=" * 80)


def main():
    """Main execution"""
    try:
        # Run comprehensive comparison
        run_all_tests()
        
        # Demonstrate specific use cases
        demonstrate_specific_use_case()
        demonstrate_batch_predictions()
        
        print("\n\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("""
This example demonstrated:

1. All 4 test approaches with different modeling strategies
2. Comparison between original and filtered datasets
3. Ensemble prediction for robust estimation
4. Production use case with uncertainty quantification
5. Batch predictions for scaling analysis

For more information, see:
- README.md for overview
- QUICKSTART.md for quick usage guide
- Individual test READMEs for detailed documentation
        """)
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError running example: {e}")
        print("\nMake sure you have installed all dependencies:")
        print("  pip install torch xgboost numpy scikit-learn joblib")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
