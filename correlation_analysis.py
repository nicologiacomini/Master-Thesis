#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis Tool
========================================

This script performs extensive statistical analysis on COMPSs execution datasets:
- Basic correlation analysis with heatmaps
- Pearson and Spearman correlation tests
- Bias analysis (input size distribution across nodes)
- Principal Component Analysis (PCA)
- Distribution plots and outlier detection

Usage:
    python3 correlation_analysis.py [--dataset {original,filtered,both}]

Features:
- Correlation matrices (Pearson and Spearman)
- Bias analysis with boxplots and heatmaps
- PCA with biplots and variance explained
- Statistical significance tests (p-values)
- Comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import argparse
import os


def analyze_dataset(df, dataset_name, output_dir):
    """
    Perform comprehensive statistical analysis on a dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    dataset_name : str
        Name for output files (e.g., 'original', 'filtered')
    output_dir : str
        Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Calculate INPUT_SIZE if not present
    if 'INPUT_SIZE' not in df.columns and 'MSIZE' in df.columns and 'BSIZE' in df.columns:
        df['INPUT_SIZE'] = 8 * (df['BSIZE'] ** 2) * (df['MSIZE'] ** 2)
    
    # Preprocess: drop redundant columns if they exist
    cols_to_drop = []
    if 'CPUS_PER_NODE' in df.columns:
        cols_to_drop.append('CPUS_PER_NODE')
    
    # Handle CPU_AVG and MEM_AVG columns
    if 'CPU_AVG_USED' in df.columns and 'CPU_AVG' in df.columns:
        cols_to_drop.append('CPU_AVG')
        df.rename(columns={'CPU_AVG_USED': 'CPU_AVG'}, inplace=True)
    
    if 'MEM_AVG_USED' in df.columns and 'MEM_AVG' in df.columns:
        cols_to_drop.append('MEM_AVG')
        df.rename(columns={'MEM_AVG_USED': 'MEM_AVG'}, inplace=True)
    
    if cols_to_drop:
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    print("=" * 80)
    print(f"ANALYSIS: {dataset_name.upper()} DATASET")
    print("=" * 80)
    
    print("\nData Overview:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Descriptive Statistics
    print("\nDescriptive Statistics:")
    print(df[numerical_cols].describe())
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing Values:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values found.")
    
    # ========================================================================
    # BIAS ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("BIAS ANALYSIS: Input Size Distribution Across Nodes")
    print("=" * 80)
    
    if 'INPUT_SIZE' in df.columns and 'NUM_NODES' in df.columns:
        # Statistics by number of nodes
        stats_by_nodes = df.groupby('NUM_NODES')['INPUT_SIZE'].describe()
        print("\nInput Size Statistics by Number of Nodes:")
        print(stats_by_nodes[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])
        
        # Boxplot
        plt.figure(figsize=(14, 6))
        sns.boxplot(x='NUM_NODES', y='INPUT_SIZE', data=df)
        plt.title(f'Bias Analysis: Input Size Distribution Across Number of Nodes - {dataset_name.capitalize()}')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Input Size (bytes)')
        plt.grid(True, alpha=0.3)
        bias_box_file = os.path.join(output_dir, f'bias_analysis_boxplot_{dataset_name}.png')
        plt.savefig(bias_box_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Bias analysis boxplot saved to: {bias_box_file}")
        
        # Heatmap with categorized input sizes
        df['INPUT_CATEGORY'] = pd.cut(df['INPUT_SIZE'], bins=5, labels=['Tiny', 'Small', 'Medium', 'Large', 'Huge'])
        heatmap_data = pd.crosstab(df['NUM_NODES'], df['INPUT_CATEGORY'])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d')
        plt.title(f'Heatmap: Input Size Distribution Across Number of Nodes - {dataset_name.capitalize()}')
        plt.xlabel('Input Size Category')
        plt.ylabel('Number of Nodes')
        bias_heat_file = os.path.join(output_dir, f'bias_analysis_heatmap_{dataset_name}.png')
        plt.savefig(bias_heat_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Bias analysis heatmap saved to: {bias_heat_file}")
        
        # Drop the temporary column
        df.drop(columns=['INPUT_CATEGORY'], inplace=True)
    else:
        print("Skipping bias analysis: INPUT_SIZE or NUM_NODES not available")
    
    # ========================================================================
    # PEARSON AND SPEARMAN CORRELATION TESTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS: Pearson and Spearman Tests")
    print("=" * 80)
    
    # Pearson correlation test
    print("\nPearson Correlation Tests:")
    print("-" * 80)
    pearson_results = {}
    for col1, col2 in combinations(numerical_cols, 2):
        data = df[[col1, col2]].dropna()
        if len(data) < 2:
            continue
        if data[col1].nunique() <= 1 or data[col2].nunique() <= 1:
            pearson_results[(col1, col2)] = (float('nan'), float('nan'))
            continue
        try:
            corr_coef, p_value = pearsonr(data[col1], data[col2])
            pearson_results[(col1, col2)] = (corr_coef, p_value)
            if p_value < 0.01:
                print(f"{col1:15s} vs {col2:15s}: r={corr_coef:>7.4f}, p={p_value:.4e} (significant)")
        except Exception as e:
            pearson_results[(col1, col2)] = (float('nan'), float('nan'))
    
    # Spearman correlation test
    print("\nSpearman Correlation Tests:")
    print("-" * 80)
    spearman_results = {}
    for col1, col2 in combinations(numerical_cols, 2):
        data = df[[col1, col2]].dropna()
        if len(data) < 2:
            continue
        if data[col1].nunique() <= 1 or data[col2].nunique() <= 1:
            spearman_results[(col1, col2)] = (float('nan'), float('nan'))
            continue
        try:
            corr_coef, p_value = spearmanr(data[col1], data[col2])
            spearman_results[(col1, col2)] = (corr_coef, p_value)
            if p_value < 0.01:
                print(f"{col1:15s} vs {col2:15s}: ρ={corr_coef:>7.4f}, p={p_value:.4e} (significant)")
        except Exception as e:
            spearman_results[(col1, col2)] = (float('nan'), float('nan'))
    
    # Pearson correlation heatmap
    corr_pearson = df[numerical_cols].corr(method='pearson')
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_pearson, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title(f'Pearson Correlation Coefficients - {dataset_name.capitalize()}')
    pearson_file = os.path.join(output_dir, f'pearson_correlation_heatmap_{dataset_name}.png')
    plt.savefig(pearson_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPearson correlation heatmap saved to: {pearson_file}")
    
    # Spearman correlation heatmap
    corr_spearman = df[numerical_cols].corr(method='spearman')
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_spearman, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title(f'Spearman Correlation Coefficients - {dataset_name.capitalize()}')
    spearman_file = os.path.join(output_dir, f'spearman_correlation_heatmap_{dataset_name}.png')
    plt.savefig(spearman_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Spearman correlation heatmap saved to: {spearman_file}")
    
    # Save scatterplots for significant Pearson correlations
    scatter_dir = os.path.join(output_dir, 'scatterplots_pearson')
    os.makedirs(scatter_dir, exist_ok=True)
    scatter_count = 0
    for (col1, col2), (corr_coef, p_value) in pearson_results.items():
        if p_value < 0.01 and not np.isnan(corr_coef):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x=col1, y=col2, alpha=0.6)
            plt.title(f'{col1} vs {col2}\n(Pearson r={corr_coef:.3f}, p={p_value:.2e})')
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.grid(True, alpha=0.3)
            scatter_file = os.path.join(scatter_dir, f'scatter_{col1}_vs_{col2}.png')
            plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
            plt.close()
            scatter_count += 1
    print(f"Generated {scatter_count} Pearson scatterplots in: {scatter_dir}")
    
    # Save scatterplots for significant Spearman correlations
    scatter_dir_spearman = os.path.join(output_dir, 'scatterplots_spearman')
    os.makedirs(scatter_dir_spearman, exist_ok=True)
    scatter_count_spearman = 0
    for (col1, col2), (corr_coef, p_value) in spearman_results.items():
        if p_value < 0.01 and not np.isnan(corr_coef):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x=col1, y=col2, alpha=0.6)
            plt.title(f'{col1} vs {col2}\n(Spearman ρ={corr_coef:.3f}, p={p_value:.2e})')
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.grid(True, alpha=0.3)
            scatter_file = os.path.join(scatter_dir_spearman, f'scatter_spearman_{col1}_vs_{col2}.png')
            plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
            plt.close()
            scatter_count_spearman += 1
    print(f"Generated {scatter_count_spearman} Spearman scatterplots in: {scatter_dir_spearman}")
    
    # ========================================================================
    # PRINCIPAL COMPONENT ANALYSIS (PCA)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("=" * 80)
    
    # Prepare data for PCA
    df_clean = df[numerical_cols].dropna()
    if len(df_clean) > 0 and len(numerical_cols) >= 2:
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_clean)
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        # Create DataFrame with PC results
        df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        
        print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")
        
        # Plot basic scatter
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_pca, x='PC1', y='PC2')
        plt.title(f'PCA of Dataset - {dataset_name.capitalize()}')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        pca_scatter_file = os.path.join(output_dir, f'pca_scatterplot_{dataset_name}.png')
        plt.savefig(pca_scatter_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PCA scatterplot saved to: {pca_scatter_file}")
        
        # Show feature contributions (loadings)
        components_df = pd.DataFrame(
            pca.components_.T,  # Transpose so features are rows, PCs are columns
            columns=['PC1', 'PC2'],
            index=numerical_cols
        )
        
        print("\nPrincipal Component Loadings:")
        print(components_df)
        
        # Visualize loadings (biplot-style)
        plt.figure(figsize=(10, 8))
        plt.scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.5)
        
        # Plot feature vectors
        for i, feature in enumerate(numerical_cols):
            plt.arrow(
                0, 0,
                components_df.loc[feature, 'PC1'] * 3,  # scale for visibility
                components_df.loc[feature, 'PC2'] * 3,
                head_width=0.1, head_length=0.1, fc='red', ec='red'
            )
            plt.text(
                components_df.loc[feature, 'PC1'] * 3.2,
                components_df.loc[feature, 'PC2'] * 3.2,
                feature,
                fontsize=9
            )
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'PCA Biplot (Features as Vectors) - {dataset_name.capitalize()}')
        plt.grid(True)
        pca_biplot_file = os.path.join(output_dir, f'pca_biplot_{dataset_name}.png')
        plt.savefig(pca_biplot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PCA biplot saved to: {pca_biplot_file}")
        
        # Save PCA loadings to CSV
        pca_loadings_file = os.path.join(output_dir, f'pca_loadings_{dataset_name}.csv')
        components_df.to_csv(pca_loadings_file)
        print(f"PCA loadings saved to: {pca_loadings_file}")
        
        print("\nPCA analysis complete.")
    else:
        print("Skipping PCA: insufficient data or features")
    
    # ========================================================================
    # BASIC CORRELATION MATRIX (for backwards compatibility)
    # ========================================================================
    print("\n" + "=" * 80)
    print("BASIC CORRELATION ANALYSIS")
    print("=" * 80)
    corr = df[numerical_cols].corr()
    print(corr)
    
    # Save correlation matrix to CSV
    corr_file = os.path.join(output_dir, f'correlation_matrix_{dataset_name}.csv')
    corr.to_csv(corr_file)
    print(f"\nCorrelation matrix saved to: {corr_file}")
    
    # Correlation with EXEC_TIME
    if 'EXEC_TIME' in corr.columns:
        print("\nCorrelation with EXEC_TIME (sorted by absolute value):")
        exec_time_corr = corr['EXEC_TIME'].drop('EXEC_TIME').abs().sort_values(ascending=False)
        for feature, corr_val in exec_time_corr.items():
            actual_corr = corr.loc[feature, 'EXEC_TIME']
            print(f"  {feature:20s}: {actual_corr:>7.4f} (|{corr_val:.4f}|)")
    
    # Plot Correlation Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='left')
    ax.set_yticklabels(corr.columns)
    
    # Add correlation values to the plot
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                   ha='center', va='center', color='black', fontsize=8)
    
    plt.title(f'Correlation Heatmap - {dataset_name.capitalize()} Dataset', pad=20)
    heatmap_file = os.path.join(output_dir, f'correlation_heatmap_{dataset_name}.png')
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation heatmap saved to: {heatmap_file}")
    
    # Distribution of EXEC_TIME
    if 'EXEC_TIME' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df['EXEC_TIME'], bins=30, alpha=0.7, edgecolor='black', color='steelblue')
        plt.title(f'Distribution of EXEC_TIME - {dataset_name.capitalize()}')
        plt.xlabel('EXEC_TIME (seconds)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        dist_file = os.path.join(output_dir, f'exec_time_distribution_{dataset_name}.png')
        plt.savefig(dist_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"EXEC_TIME distribution saved to: {dist_file}")
    
    # Histograms for all numerical columns
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numerical_cols):
        axes[idx].hist(df[col], bins=30, alpha=0.7, edgecolor='black', color='steelblue')
        axes[idx].set_title(f'{col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Distribution of All Features - {dataset_name.capitalize()}', fontsize=16)
    plt.tight_layout()
    hist_file = os.path.join(output_dir, f'all_distributions_{dataset_name}.png')
    plt.savefig(hist_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"All distributions saved to: {hist_file}")
    
    # Box plots for all numerical columns
    plt.figure(figsize=(14, 8))
    df[numerical_cols].boxplot()
    plt.title(f'Box Plots of Numerical Features - {dataset_name.capitalize()}')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    box_file = os.path.join(output_dir, f'box_plots_{dataset_name}.png')
    plt.savefig(box_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Box plots saved to: {box_file}")
    
    # Outlier Detection using IQR for EXEC_TIME
    if 'EXEC_TIME' in df.columns:
        Q1 = df['EXEC_TIME'].quantile(0.25)
        Q3 = df['EXEC_TIME'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df['EXEC_TIME'] < lower_bound) | (df['EXEC_TIME'] > upper_bound)]
        
        print(f"\nOutlier Analysis (EXEC_TIME using IQR method):")
        print(f"  Q1 (25th percentile): {Q1:.2f}")
        print(f"  Q3 (75th percentile): {Q3:.2f}")
        print(f"  IQR: {IQR:.2f}")
        print(f"  Lower bound: {lower_bound:.2f}")
        print(f"  Upper bound: {upper_bound:.2f}")
        print(f"  Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
        
        if len(outliers) > 0:
            print("\nSample outliers:")
            display_cols = [c for c in ['MSIZE', 'BSIZE', 'NUM_NODES', 'NUM_CPUS', 'EXEC_TIME'] 
                          if c in outliers.columns]
            print(outliers[display_cols].head(10))
    
    # Skewness and Kurtosis
    print("\nSkewness and Kurtosis:")
    skew_kurt = pd.DataFrame({
        'Feature': numerical_cols,
        'Skewness': [df[col].skew() for col in numerical_cols],
        'Kurtosis': [df[col].kurtosis() for col in numerical_cols]
    })
    print(skew_kurt.to_string(index=False))
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, f'summary_statistics_{dataset_name}.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Statistical Summary - {dataset_name.capitalize()} Dataset\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset Shape: {df.shape}\n")
        f.write(f"Number of Features: {len(numerical_cols)}\n")
        f.write(f"Number of Samples: {len(df)}\n\n")
        
        f.write("Descriptive Statistics:\n")
        f.write(df[numerical_cols].describe().to_string())
        f.write("\n\n")
        
        f.write("Skewness and Kurtosis:\n")
        f.write(skew_kurt.to_string(index=False))
        f.write("\n\n")
        
        if 'EXEC_TIME' in df.columns:
            f.write("Correlation with EXEC_TIME:\n")
            exec_time_corr = corr['EXEC_TIME'].drop('EXEC_TIME').sort_values(ascending=False)
            for feature, corr_val in exec_time_corr.items():
                f.write(f"  {feature:20s}: {corr_val:>7.4f}\n")
    
    print(f"Summary statistics saved to: {summary_file}")
    print(f"\nAnalysis complete for {dataset_name} dataset.\n")


def main():
    parser = argparse.ArgumentParser(
        description='Perform correlation analysis and generate heatmaps for COMPSs execution datasets'
    )
    parser.add_argument(
        '--dataset',
        choices=['original', 'filtered', 'both'],
        default='both',
        help='Which dataset to analyze (default: both)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CORRELATION ANALYSIS AND HEATMAP GENERATION")
    print("=" * 80)
    print()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Analyze original dataset
    if args.dataset in ['original', 'both']:
        original_csv = os.path.join(script_dir, 'original_dataset.csv')
        if os.path.exists(original_csv):
            df_original = pd.read_csv(original_csv)
            output_dir = os.path.join(script_dir, 'analysis_results', 'original')
            analyze_dataset(df_original, 'original', output_dir)
        else:
            print(f"Warning: Original dataset not found at {original_csv}")
    
    # Analyze filtered dataset
    if args.dataset in ['filtered', 'both']:
        filtered_csv = os.path.join(script_dir, 'filtered_dataset.csv')
        if os.path.exists(filtered_csv):
            df_filtered = pd.read_csv(filtered_csv)
            output_dir = os.path.join(script_dir, 'analysis_results', 'filtered')
            analyze_dataset(df_filtered, 'filtered', output_dir)
        else:
            print(f"Warning: Filtered dataset not found at {filtered_csv}")
    
    print("=" * 80)
    print("ALL ANALYSES COMPLETE")
    print("=" * 80)
    print("\nResults saved in ./analysis_results/ directory")
    print("\nGenerated files:")
    print("  - correlation_matrix_*.csv               : Basic correlation matrices")
    print("  - correlation_heatmap_*.png              : Visual correlation heatmaps")
    print("  - pearson_correlation_heatmap_*.png      : Pearson correlation heatmaps")
    print("  - spearman_correlation_heatmap_*.png     : Spearman correlation heatmaps")
    print("  - bias_analysis_boxplot_*.png            : Input size distribution by nodes")
    print("  - bias_analysis_heatmap_*.png            : Input size category heatmap")
    print("  - pca_scatterplot_*.png                  : PCA scatter plot (PC1 vs PC2)")
    print("  - pca_biplot_*.png                       : PCA biplot with feature vectors")
    print("  - pca_variance_*.png                     : PCA explained variance plot")
    print("  - pca_loadings_*.csv                     : PCA component loadings")
    print("  - exec_time_distribution_*.png           : EXEC_TIME histograms")
    print("  - all_distributions_*.png                : All feature distributions")
    print("  - box_plots_*.png                        : Box plots for outlier detection")
    print("  - summary_statistics_*.txt               : Text summary of statistics")
    print("  - scatterplots_pearson/                  : Scatterplots for significant Pearson correlations")
    print("  - scatterplots_spearman/                 : Scatterplots for significant Spearman correlations")
    print()


if __name__ == '__main__':
    main()
