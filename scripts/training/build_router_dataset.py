import os
import yaml
import argparse
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "router_config_v1.yaml"
COST_MATRIX_PATH = BASE_DIR / "configs" / "cost_matrix_v1.yaml"
SPLITS_DIR = BASE_DIR / "data" / "splits"

def build_dataset_for_split(split_name, cost_matrix):
    small_path = SPLITS_DIR / f"raw_features_small_{split_name}.parquet"
    large_path = SPLITS_DIR / f"raw_features_large_{split_name}.parquet"
    out_path = SPLITS_DIR / f"router_dataset_v1_{split_name}.parquet"
    
    if not os.path.exists(small_path):
        print(f"️ Skip {split_name}: File not found {small_path.name}")
        return
        
    if not os.path.exists(large_path):
        print(f"️ Skip {split_name}: File not found {large_path.name}")
        return
        
    print(f"\\n Processing split: {split_name}")
    
    # Load 2 raw feature tables
    df_small = pd.read_parquet(small_path)
    df_large = pd.read_parquet(large_path)
    
    # Rename necessary columns to prevent conflict during merge
    # Keep small model features as main features for router
    df_large = df_large[['sample_id', 'prediction']].rename(
        columns={'prediction': 'prediction_large'}
    )
    
    # Merge theo sample_id
    df_merged = pd.merge(df_small, df_large, on='sample_id', how='inner')
    print(f" Successfully merged. Size (Rows): {len(df_merged)}")
    
    # Calculate Cost
    def calc_cost(row, pred_col):
        gold = str(row['gold_label']).lower().strip()
        pred = str(row[pred_col]).lower().strip()
        
        # Fallback if model generates invalid outputs
        if pred not in ['safe', 'controversial', 'unsafe']:
            pred = 'unsafe' # Penalize model if it generates incorrect format
            
        try:
            return cost_matrix[gold][pred]
        except KeyError:
            print(f"Error: Cost not found for Gold: {gold} -> Pred: {pred}")
            return 100 # Heavy penalty for unexpected errors
            
    df_merged['cost06'] = df_merged.apply(lambda r: calc_cost(r, 'prediction'), axis=1)
    df_merged['cost4'] = df_merged.apply(lambda r: calc_cost(r, 'prediction_large'), axis=1)
    
    # Labeling Rule: Only route to 4B when 4B cost is strictly LESS THAN 0.6B cost
    df_merged['route'] = (df_merged['cost4'] < df_merged['cost06']).astype(int)
    
    # Rename 0.6B prediction to prediction_small for clarity
    df_merged = df_merged.rename(columns={'prediction': 'prediction_small'})
    
    # Print basic distribution
    print(f" Route distribution on {split_name}:")
    print(df_merged['route'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
    
    print(f" Average Cost 0.6B: {df_merged['cost06'].mean():.3f}")
    print(f" Average Cost 4B:   {df_merged['cost4'].mean():.3f}")
    
    # Save to new file
    df_merged.to_parquet(out_path, index=False)
    print(f" Saved Router Dataset: {out_path.name}")

def main():
    print(" Starting to Build Router Dataset")
    
    # Load cost matrix
    if not os.path.exists(COST_MATRIX_PATH):
        print(f" Not found Cost Matrix at {COST_MATRIX_PATH}")
        return
        
    with open(COST_MATRIX_PATH, "r", encoding="utf-8") as f:
        cost_matrix = yaml.safe_load(f)
        
    print(f" Loaded Cost Matrix v1")
    
    # Apply to both Train and Test
    build_dataset_for_split("train", cost_matrix)
    build_dataset_for_split("test", cost_matrix)

if __name__ == "__main__":
    main()
