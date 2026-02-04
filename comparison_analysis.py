import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os


# Load the original and filtered datasets
print("=" * 80)
print("MODEL COMPARISON ANALYSIS - ALL TESTS")
print("=" * 80)

df_original = pd.read_csv('original_dataset.csv')
df_filtered = pd.read_csv('filtered_dataset.csv')

print(f"\nOriginal dataset size: {len(df_original)} records")
print(f"Filtered dataset size: {len(df_filtered)} records")

# Test configurations
tests = {
    'test_1': 'Test 1: Input Size-Based Prediction',
    'test_2': 'Test 2: Matrix Parameters-Based Prediction', 
    'test_3': 'Test 3: USL with MSIZE Parameter',
    'test_4': 'Test 4: K-Fold Cross-Validation'
}

datasets = {
    'original': df_original,
    'filtered': df_filtered
}

# Store all results
all_results = []

# Run predictions for each test and dataset
for test_name, test_desc in tests.items():
    print(f"\n{'=' * 80}")
    print(f"{test_desc}")
    print(f"{'=' * 80}")
    
    for dataset_name, df in datasets.items():
        print(f"\n  Processing {dataset_name} dataset...")
        
        # Import the prediction functions from each test
        test_path = os.path.join(os.path.dirname(__file__), test_name, dataset_name + '_dataset')
        
        # Run FFNN predictions
        try:
            ffnn_path = os.path.join(test_path, 'ffnn')
            sys.path.insert(0, ffnn_path)
            import ffnn_model
            
            ffnn_predictions = []
            for idx, row in df.iterrows():
                try:
                    pred = ffnn_model.run_ffnn(
                        int(row['NUM_NODES']),
                        int(row['MSIZE']),
                        int(row['BSIZE'])
                    )
                    ffnn_predictions.append(pred)
                except Exception as e:
                    ffnn_predictions.append(np.nan)
            
            # Remove from path to avoid conflicts
            sys.path.pop(0)
            sys.modules.pop('ffnn_model', None)
            
            print(f"    ✓ FFNN predictions completed: {len([p for p in ffnn_predictions if not pd.isna(p)])} valid")
            
        except Exception as e:
            print(f"    ✗ FFNN failed: {str(e)}")
            ffnn_predictions = [np.nan] * len(df)
        
        # Run XGBoost predictions
        try:
            xgboost_path = os.path.join(test_path, 'xgboost')
            sys.path.insert(0, xgboost_path)
            import xgboost_model
            
            xgboost_predictions = []
            for idx, row in df.iterrows():
                try:
                    pred = xgboost_model.run_xgboost(
                        int(row['NUM_NODES']),
                        int(row['MSIZE']),
                        int(row['BSIZE'])
                    )
                    xgboost_predictions.append(pred)
                except Exception as e:
                    xgboost_predictions.append(np.nan)
            
            # Remove from path to avoid conflicts
            sys.path.pop(0)
            sys.modules.pop('xgboost_model', None)
            
            print(f"    ✓ XGBoost predictions completed: {len([p for p in xgboost_predictions if not pd.isna(p)])} valid")
            
        except Exception as e:
            print(f"    ✗ XGBoost failed: {str(e)}")
            xgboost_predictions = [np.nan] * len(df)
        
        # Store results
        result_df = df.copy()
        result_df[f'{test_name}_FFNN_{dataset_name.upper()}'] = ffnn_predictions
        result_df[f'{test_name}_XGBOOST_{dataset_name.upper()}'] = xgboost_predictions
        all_results.append(result_df)

# Combine all results
print(f"\n{'=' * 80}")
print("COMBINING RESULTS")
print(f"{'=' * 80}")

# Get the base dataframe (use original as base)
final_df = df_original.copy()

# Add all prediction columns from all tests
for result_df in all_results:
    for col in result_df.columns:
        if col.startswith('test_'):
            final_df[col] = result_df[col]

# Save combined results
output_file = 'all_tests_predictions.csv'
final_df.to_csv(output_file, index=False)
print(f"\n✓ All predictions saved to: {output_file}")

# Generate comparison plots
print(f"\n{'=' * 80}")
print("GENERATING COMPARISON PLOTS")
print(f"{'=' * 80}")

# Get model columns (all test predictions)
model_names = [col for col in final_df.columns if col.startswith('test_')]

for model in model_names:
    print(f"\nGenerating plot for: {model}")
    
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10, 6))

    # compute percent error (handle EXEC_TIME == 0)
    pct_err = np.where(final_df['EXEC_TIME'] != 0, (final_df[model] - final_df['EXEC_TIME']) / final_df['EXEC_TIME'] * 100, np.nan)
    final_df[f'{model}_ERROR_%'] = pct_err

    # label: <=10% -> Correct, >10% and prediction lower -> Unsafe, >10% and prediction higher -> Safe
    conditions = [np.abs(pct_err) <= 10, pct_err < -10, pct_err > 10]
    choices = ['Correct', 'Unsafe (underestimation)', 'Safe (overestimation)']
    final_df[f'{model}_SAFETY'] = np.select(conditions, choices, default='Unknown')
    
    # Calculate statistics
    correct_pct = (final_df[f'{model}_SAFETY'] == 'Correct').sum() / len(final_df) * 100
    safe_pct = (final_df[f'{model}_SAFETY'] == 'Safe (overestimation)').sum() / len(final_df) * 100
    unsafe_pct = (final_df[f'{model}_SAFETY'] == 'Unsafe (underestimation)').sum() / len(final_df) * 100
    
    print(f"  Correct: {correct_pct:.2f}% | Safe: {safe_pct:.2f}% | Unsafe: {unsafe_pct:.2f}%")

    sns.scatterplot(
        data=final_df,
        x='EXEC_TIME',
        y=model,
        hue=f'{model}_SAFETY',
        palette={'Correct':'green', 'Safe (overestimation)': 'darkorange', 'Unsafe (underestimation)': 'red', 'Unknown': 'gray'},
        alpha=0.6,
        s=70,
    )

    # plot y=x line
    max_time = max(final_df['EXEC_TIME'].max(), final_df[model].max())
    plt.plot([0, max_time], [0, max_time], color='blue', linestyle='--', label='Ideal Prediction (y=x)')

    plt.title(f'Accuracy and Safety of {model} Predictions')
    plt.xlabel('Real Execution Time (s)')
    plt.ylabel('Predicted Execution Time (s)')
    plt.legend()
    plt.tight_layout()
    
    # Save in comparison_results folder
    os.makedirs('comparison_results', exist_ok=True)
    output_path = os.path.join('comparison_results', f'{model}_accuracy_safety.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

print(f"\n{'=' * 80}")
print("ANALYSIS COMPLETE")
print(f"{'=' * 80}")
print(f"Results saved to: all_tests_predictions.csv")
print(f"Plots saved to: comparison_results/")
print(f"{'=' * 80}")
