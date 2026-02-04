#!/usr/bin/env python3
"""
Simple Models Comparison Analysis
Generates predictions from all simple_models and creates accuracy/safety plots
similar to comparison_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compare all simple models predictions')
parser.add_argument('--dataset', type=str, choices=['original', 'filtered', 'both'], 
                    default='both', help='Which dataset to use')
args = parser.parse_args()

print("=" * 80)
print("SIMPLE MODELS COMPARISON ANALYSIS")
print("=" * 80)

# Determine which datasets to process
datasets_to_process = []
if args.dataset in ['original', 'both']:
    datasets_to_process.append(('original', 'original_dataset.csv'))
if args.dataset in ['filtered', 'both']:
    datasets_to_process.append(('filtered', 'filtered_dataset.csv'))

# Model configurations
models_config = {
    'Linear Regression': {
        'original': 'simple_models.linear_regression_best',
        'filtered': 'simple_models.linear_regression_best_filtered'
    },
    'Polynomial Regression': {
        'original': 'simple_models.polynomial_regression_best',
        'filtered': 'simple_models.polynomial_regression_best_filtered'
    },
    'Random Forest': {
        'original': 'simple_models.random_forest_best',
        'filtered': 'simple_models.random_forest_best_filtered'
    },
    'FFNN': {
        'original': 'simple_models.ffnn_best_original',
        'filtered': 'simple_models.ffnn_best_filtered'
    },
    'XGBoost': {
        'original': 'simple_models.xgboost_best_original',
        'filtered': 'simple_models.xgboost_best_filtered'
    }
}

# Process each dataset
for dataset_name, dataset_file in datasets_to_process:
    print(f"\n{'=' * 80}")
    print(f"Processing {dataset_name.upper()} dataset: {dataset_file}")
    print(f"{'=' * 80}")
    
    # Load dataset
    df = pd.read_csv(dataset_file)
    print(f"Dataset size: {len(df)} records")
    
    # Create results dataframe
    results_df = df.copy()
    
    # Run predictions for each model
    for model_name, model_modules in models_config.items():
        module_name = model_modules[dataset_name]
        
        print(f"\n  Running {model_name} ({dataset_name})...")
        
        try:
            # Import the model module
            module = __import__(module_name, fromlist=['exec_prediction'])
            predict_func = getattr(module, 'exec_prediction')
            
            # Generate predictions for all rows
            predictions = []
            for idx, row in df.iterrows():
                try:
                    pred = predict_func(
                        int(row['NUM_NODES']),
                        int(row['MSIZE']),
                        int(row['BSIZE'])
                    )
                    predictions.append(pred)
                except Exception as e:
                    predictions.append(np.nan)
                
                # Progress indicator for large datasets
                if (idx + 1) % 50 == 0:
                    print(f"    Progress: {idx + 1}/{len(df)} predictions", end='\r')
            
            print(f"    ✓ Completed: {len([p for p in predictions if not pd.isna(p)])} valid predictions")
            
            # Add predictions to results
            col_name = f'{model_name}_{dataset_name.upper()}'
            results_df[col_name] = predictions
            
        except Exception as e:
            print(f"    ✗ Failed: {str(e)}")
            results_df[f'{model_name}_{dataset_name.upper()}'] = np.nan
    
    # Save results to CSV
    output_csv = f'comparison_simple_models_{dataset_name}.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Predictions saved to: {output_csv}")
    
    # Generate comparison plots
    print(f"\n{'=' * 80}")
    print(f"GENERATING COMPARISON PLOTS FOR {dataset_name.upper()} DATASET")
    print(f"{'=' * 80}")
    
    # Get model prediction columns
    model_columns = [col for col in results_df.columns if col.endswith(f'_{dataset_name.upper()}')]
    
    # Create output directory
    output_dir = f'comparison_results/simple_models_{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plot for each model
    for model_col in model_columns:
        print(f"\nGenerating plot for: {model_col}")
        
        # Filter out NaN predictions
        valid_data = results_df[~results_df[model_col].isna()].copy()
        
        if len(valid_data) == 0:
            print(f"  ⚠️  No valid predictions, skipping plot")
            continue
        
        # Set plot style
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(10, 6))
        
        # Compute percent error (handle EXEC_TIME == 0)
        pct_err = np.where(
            valid_data['EXEC_TIME'] != 0,
            (valid_data[model_col] - valid_data['EXEC_TIME']) / valid_data['EXEC_TIME'] * 100,
            np.nan
        )
        valid_data[f'{model_col}_ERROR_%'] = pct_err
        
        # Label: <=10% -> Correct, >10% and prediction lower -> Unsafe, >10% and prediction higher -> Safe
        conditions = [
            np.abs(pct_err) <= 10,
            pct_err < -10,
            pct_err > 10
        ]
        choices = ['Correct', 'Unsafe (underestimation)', 'Safe (overestimation)']
        valid_data[f'{model_col}_SAFETY'] = np.select(conditions, choices, default='Unknown')
        
        # Calculate statistics
        correct_count = (valid_data[f'{model_col}_SAFETY'] == 'Correct').sum()
        safe_count = (valid_data[f'{model_col}_SAFETY'] == 'Safe (overestimation)').sum()
        unsafe_count = (valid_data[f'{model_col}_SAFETY'] == 'Unsafe (underestimation)').sum()
        total = len(valid_data)
        
        correct_pct = (correct_count / total) * 100
        safe_pct = (safe_count / total) * 100
        unsafe_pct = (unsafe_count / total) * 100
        
        print(f"  Correct: {correct_pct:.2f}% ({correct_count}/{total})")
        print(f"  Safe: {safe_pct:.2f}% ({safe_count}/{total})")
        print(f"  Unsafe: {unsafe_pct:.2f}% ({unsafe_count}/{total})")
        
        # Create scatter plot
        sns.scatterplot(
            data=valid_data,
            x='EXEC_TIME',
            y=model_col,
            hue=f'{model_col}_SAFETY',
            palette={
                'Correct': 'green',
                'Safe (overestimation)': 'darkorange',
                'Unsafe (underestimation)': 'red',
                'Unknown': 'gray'
            },
            alpha=0.6,
            s=70,
        )
        
        # Plot y=x line (ideal prediction)
        max_time = max(valid_data['EXEC_TIME'].max(), valid_data[model_col].max())
        min_time = min(valid_data['EXEC_TIME'].min(), valid_data[model_col].min())
        plt.plot([min_time, max_time], [min_time, max_time], 
                color='blue', linestyle='--', label='Ideal Prediction (y=x)', linewidth=2)
        
        plt.title(f'Accuracy and Safety of {model_col} Predictions')
        plt.xlabel('Real Execution Time (s)')
        plt.ylabel('Predicted Execution Time (s)')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, f'{model_col}_accuracy_safety.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path}")
    
    print(f"\n{'=' * 80}")
    print(f"ANALYSIS COMPLETE FOR {dataset_name.upper()} DATASET")
    print(f"{'=' * 80}")
    print(f"Results saved to: {output_csv}")
    print(f"Plots saved to: {output_dir}/")

# Generate summary comparison if both datasets were processed
if len(datasets_to_process) == 2:
    print(f"\n{'=' * 80}")
    print("GENERATING SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    
    summary_data = []
    
    for dataset_name, _ in datasets_to_process:
        csv_file = f'comparison_simple_models_{dataset_name}.csv'
        df = pd.read_csv(csv_file)
        
        model_columns = [col for col in df.columns if col.endswith(f'_{dataset_name.upper()}')]
        
        for model_col in model_columns:
            valid_data = df[~df[model_col].isna()].copy()
            
            if len(valid_data) == 0:
                continue
            
            # Calculate metrics
            pct_err = np.where(
                valid_data['EXEC_TIME'] != 0,
                (valid_data[model_col] - valid_data['EXEC_TIME']) / valid_data['EXEC_TIME'] * 100,
                np.nan
            )
            
            mae = np.mean(np.abs(valid_data[model_col] - valid_data['EXEC_TIME']))
            rmse = np.sqrt(np.mean((valid_data[model_col] - valid_data['EXEC_TIME']) ** 2))
            mape = np.mean(np.abs(pct_err))
            
            correct_count = np.sum(np.abs(pct_err) <= 10)
            safe_count = np.sum(pct_err > 10)
            unsafe_count = np.sum(pct_err < -10)
            
            summary_data.append({
                'Model': model_col,
                'Dataset': dataset_name,
                'Total Predictions': len(valid_data),
                'Correct (%)': (correct_count / len(valid_data)) * 100,
                'Safe (%)': (safe_count / len(valid_data)) * 100,
                'Unsafe (%)': (unsafe_count / len(valid_data)) * 100,
                'MAE (s)': mae,
                'RMSE (s)': rmse,
                'MAPE (%)': mape
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = 'comparison_simple_models_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\n✓ Summary statistics saved to: {summary_csv}")
    print("\nSummary Preview:")
    print(summary_df.to_string(index=False))

print(f"\n{'=' * 80}")
print("ALL ANALYSIS COMPLETE")
print(f"{'=' * 80}")
