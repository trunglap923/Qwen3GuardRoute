import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def verify_parquet(file_path):
    print(f"\\n{'='*50}")
    print(f" Bt u kim tra (Verify): {Path(file_path).name}")
    print(f"{'='*50}")
    
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return False
        
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f" Error reading Parquet: {e}")
        return False
        
    print(f" c to cng. S lng samples (Rows): {len(df)}")
    print(f" S lng ct (Columns): {len(df.columns)}")
    
    # 1. Check Duplicate Sample ID
    if 'sample_id' in df.columns:
        dups = df['sample_id'].duplicated().sum()
        if dups > 0:
            print(f" WARNING: Detected {dups} duplicate sample_id!")
        else:
            print(f" No duplicate sample_id (Unique: {len(df['sample_id'].unique())}).")
    else:
        print(f"️ Cnh bo: Not found ct 'sample_id'.")
        
    # 2. Check NaNs
    nans = df.isna().sum()
    cols_with_nans = nans[nans > 0]
    if not cols_with_nans.empty:
        print(f" WARNING: Detected Missing Values (NaN):")
        for col, count in cols_with_nans.items():
            print(f"   - Column '{col}': {count} NaNs")
    else:
        print(f" D liu sch, khng have Missing Values (NaN).")
        
    # 3. Check Hidden Dimension
    if 'hidden' in df.columns:
        hidden_sample = df['hidden'].iloc[0]
        # In case it's a list or numpy array
        dim = len(hidden_sample)
        print(f" Vector size 'hidden' (Dimension): {dim}")
        
        # Check dimension consistency
        dim_mismatches = df['hidden'].apply(len) != dim
        if dim_mismatches.sum() > 0:
            print(f" WARNING: Detected {dim_mismatches.sum()} mẫu có dimension khác {dim}!")
        else:
            print(f" 100% samples have cng kch thc vector hidden.")
    else:
        print(f"️ No column 'hidden' in file ny.")
        
    # 4. Check Distribution
    print("\\n DISTRIBUTION STATISTICS:")
    
    if 'gold_label' in df.columns:
        print(" - Distribution 'gold_label':")
        print(df['gold_label'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
        
    if 'prediction' in df.columns:
        print("\\n - Distribution 'prediction':")
        print(df['prediction'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
        
    if 'route' in df.columns:
        print("\\n - Distribution 'route' (0: Keep, 1: Route to 4B):")
        counts = df['route'].value_counts()
        pcts = df['route'].value_counts(normalize=True).mul(100).round(1)
        for val in counts.index:
            label = "Route_to_4B" if val == 1 else "Keep_0.6B"
            print(f"   * {label} ({val}): {counts[val]} mẫu ({pcts[val]}%)")
            
    # 5. Print Metadata if available
    meta_cols = ['model_name', 'hidden_type', 'feature_version', 'extract_time']
    found_meta = [c for c in meta_cols if c in df.columns]
    if found_meta:
        print("\\n EXTRACTED METADATA:")
        for col in found_meta:
            # Get first value assuming homogeneous metadata for the entire file
            val = df[col].iloc[0]
            print(f" - {col}: {val}")

    print(f"{'='*50}\\n")
    return True

def main():
    parser = argparse.ArgumentParser(description="Kiểm tra tính toàn vẹn của dữ liệu Parquet")
    parser.add_argument("--file", type=str, help="Đường dẫn đến file parquet cần verify. Nếu không truyền, sẽ verify tất cả file parquet trong data/splits/")
    args = parser.parse_args()
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    SPLITS_DIR = BASE_DIR / "data" / "splits"
    
    if args.file:
        verify_parquet(args.file)
    else:
        print(f"Start scanning directory: {SPLITS_DIR}")
        files = list(SPLITS_DIR.glob("*.parquet"))
        if not files:
            print("File not found parquet no.")
        for f in files:
            verify_parquet(str(f))

if __name__ == "__main__":
    main()
