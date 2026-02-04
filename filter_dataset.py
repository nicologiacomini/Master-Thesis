#!/usr/bin/env python3
"""
Filter Dataset Script
This script filters the original dataset to create a filtered version
based on INPUT_SIZE threshold to ensure more homogeneous data.
"""

import pandas as pd

# File paths
ORIGINAL_FILE = 'original_dataset.csv'
FILTERED_FILE = 'filtered_dataset.csv'

# Filtering threshold (from your XGBoost script)
INPUT_SIZE_THRESHOLD = 4632608768

def main():
    print("=" * 70)
    print("Dataset Filtering Script")
    print("=" * 70)
    
    # Load original dataset
    print(f"\nLoading original dataset from: {ORIGINAL_FILE}")
    df = pd.read_csv(ORIGINAL_FILE)
    print(f"Original dataset size: {len(df)} entries")
    
    # Calculate INPUT_SIZE
    print("\nCalculating INPUT_SIZE for each entry...")
    df['INPUT_SIZE'] = (4 * (df['BSIZE'] ** 2)) * (df['MSIZE'] ** 2) * 2
    
    # Show INPUT_SIZE statistics before filtering
    print(f"\nINPUT_SIZE statistics (before filtering):")
    print(f"  Min: {df['INPUT_SIZE'].min():,.0f} bytes")
    print(f"  Max: {df['INPUT_SIZE'].max():,.0f} bytes")
    print(f"  Mean: {df['INPUT_SIZE'].mean():,.0f} bytes")
    print(f"  Median: {df['INPUT_SIZE'].median():,.0f} bytes")
    print(f"  Threshold: {INPUT_SIZE_THRESHOLD:,.0f} bytes")
    
    # Count entries above threshold
    entries_above_threshold = len(df[df['INPUT_SIZE'] > INPUT_SIZE_THRESHOLD])
    print(f"\nEntries above threshold: {entries_above_threshold} ({entries_above_threshold/len(df)*100:.1f}%)")
    
    # Apply filter
    print(f"\nApplying filter: INPUT_SIZE <= {INPUT_SIZE_THRESHOLD:,.0f}")
    df_filtered = df[df['INPUT_SIZE'] <= INPUT_SIZE_THRESHOLD].copy()
    
    # Remove the INPUT_SIZE column before saving (it's calculated on-the-fly in models)
    df_filtered = df_filtered.drop('INPUT_SIZE', axis=1)
    
    print(f"Filtered dataset size: {len(df_filtered)} entries")
    print(f"Removed: {len(df) - len(df_filtered)} entries")
    
    # Show filtered dataset statistics
    print(f"\nFiltered dataset statistics:")
    print(f"  NUM_NODES range: {df_filtered['NUM_NODES'].min()} - {df_filtered['NUM_NODES'].max()}")
    print(f"  MSIZE range: {df_filtered['MSIZE'].min()} - {df_filtered['MSIZE'].max()}")
    print(f"  BSIZE range: {df_filtered['BSIZE'].min()} - {df_filtered['BSIZE'].max()}")
    print(f"  EXEC_TIME range: {df_filtered['EXEC_TIME'].min():.2f}s - {df_filtered['EXEC_TIME'].max():.2f}s")
    
    # Save filtered dataset
    print(f"\nSaving filtered dataset to: {FILTERED_FILE}")
    df_filtered.to_csv(FILTERED_FILE, index=False)
    
    print("\n" + "=" * 70)
    print("Filtering complete!")
    print("=" * 70)
    print(f"\nOriginal dataset: {len(df)} entries")
    print(f"Filtered dataset: {len(df_filtered)} entries")
    print(f"Reduction: {len(df) - len(df_filtered)} entries ({(len(df) - len(df_filtered))/len(df)*100:.1f}%)")
    print("\nFiltering criterion: INPUT_SIZE <= 4,632,608,768 bytes (~4.3 GB)")
    print("Purpose: Remove outliers with extremely large input sizes")
    print("         to create more homogeneous training data")
    print("=" * 70)

if __name__ == "__main__":
    main()
