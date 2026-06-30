import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def verify_parquet(file_path):
    print(f"\\n{'='*50}")
    print(f"🔍 Bắt đầu kiểm tra (Verify): {Path(file_path).name}")
    print(f"{'='*50}")
    
    if not os.path.exists(file_path):
        print(f"❌ File không tồn tại: {file_path}")
        return False
        
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"❌ Lỗi đọc Parquet: {e}")
        return False
        
    print(f"✅ Đọc thành công. Số lượng mẫu (Rows): {len(df)}")
    print(f"✅ Số lượng cột (Columns): {len(df.columns)}")
    
    # 1. Kiểm tra Duplicate Sample ID
    if 'sample_id' in df.columns:
        dups = df['sample_id'].duplicated().sum()
        if dups > 0:
            print(f"❌ CẢNH BÁO: Phát hiện {dups} sample_id trùng lặp!")
        else:
            print(f"✅ Không có sample_id trùng lặp (Unique: {len(df['sample_id'].unique())}).")
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy cột 'sample_id'.")
        
    # 2. Kiểm tra NaNs
    nans = df.isna().sum()
    cols_with_nans = nans[nans > 0]
    if not cols_with_nans.empty:
        print(f"❌ CẢNH BÁO: Phát hiện Missing Values (NaN):")
        for col, count in cols_with_nans.items():
            print(f"   - Cột '{col}': {count} NaNs")
    else:
        print(f"✅ Dữ liệu sạch, không có Missing Values (NaN).")
        
    # 3. Kiểm tra Hidden Dimension
    if 'hidden' in df.columns:
        hidden_sample = df['hidden'].iloc[0]
        # In case it's a list or numpy array
        dim = len(hidden_sample)
        print(f"✅ Kích thước vector 'hidden' (Dimension): {dim}")
        
        # Kiểm tra tính nhất quán của dimension
        dim_mismatches = df['hidden'].apply(len) != dim
        if dim_mismatches.sum() > 0:
            print(f"❌ CẢNH BÁO: Phát hiện {dim_mismatches.sum()} mẫu có dimension khác {dim}!")
        else:
            print(f"✅ 100% mẫu có cùng kích thước vector hidden.")
    else:
        print(f"ℹ️ Không có cột 'hidden' trong file này.")
        
    # 4. Kiểm tra Phân bổ (Distribution)
    print("\\n📊 THỐNG KÊ PHÂN BỔ:")
    
    if 'gold_label' in df.columns:
        print(" - Phân bổ 'gold_label':")
        print(df['gold_label'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
        
    if 'prediction' in df.columns:
        print("\\n - Phân bổ 'prediction':")
        print(df['prediction'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
        
    if 'route' in df.columns:
        print("\\n - Phân bổ 'route' (0: Giữ lại, 1: Chuyển 4B):")
        counts = df['route'].value_counts()
        pcts = df['route'].value_counts(normalize=True).mul(100).round(1)
        for val in counts.index:
            label = "Route_to_4B" if val == 1 else "Keep_0.6B"
            print(f"   * {label} ({val}): {counts[val]} mẫu ({pcts[val]}%)")
            
    # 5. In Metadata nếu có
    meta_cols = ['model_name', 'hidden_type', 'feature_version', 'extract_time']
    found_meta = [c for c in meta_cols if c in df.columns]
    if found_meta:
        print("\\n📝 METADATA TRÍCH XUẤT:")
        for col in found_meta:
            # Lấy giá trị đầu tiên giả định metadata đồng nhất cho cả file
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
        print(f"Bắt đầu quét thư mục: {SPLITS_DIR}")
        files = list(SPLITS_DIR.glob("*.parquet"))
        if not files:
            print("Không tìm thấy file parquet nào.")
        for f in files:
            verify_parquet(str(f))

if __name__ == "__main__":
    main()
