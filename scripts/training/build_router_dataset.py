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
        print(f"⚠️ Bỏ qua {split_name}: Không tìm thấy file {small_path.name}")
        return
        
    if not os.path.exists(large_path):
        print(f"⚠️ Bỏ qua {split_name}: Không tìm thấy file {large_path.name}")
        return
        
    print(f"\\n🔄 Đang xử lý tập: {split_name}")
    
    # Load 2 bảng raw features
    df_small = pd.read_parquet(small_path)
    df_large = pd.read_parquet(large_path)
    
    # Đổi tên các cột cần thiết để không bị conflict khi merge
    # Ta giữ feature của small model làm feature chính cho router
    df_large = df_large[['sample_id', 'prediction']].rename(
        columns={'prediction': 'prediction_large'}
    )
    
    # Merge theo sample_id
    df_merged = pd.merge(df_small, df_large, on='sample_id', how='inner')
    print(f"✅ Đã merge thành công. Kích thước (Rows): {len(df_merged)}")
    
    # Tính Cost
    def calc_cost(row, pred_col):
        gold = str(row['gold_label']).lower().strip()
        pred = str(row[pred_col]).lower().strip()
        
        # Fallback nếu model sinh bậy bạ
        if pred not in ['safe', 'controversial', 'unsafe']:
            pred = 'unsafe' # Phạt model nếu sinh sai định dạng
            
        try:
            return cost_matrix[gold][pred]
        except KeyError:
            print(f"Lỗi: Không tìm thấy cost cho Gold: {gold} -> Pred: {pred}")
            return 100 # Phạt nặng nếu lỗi không lường trước
            
    df_merged['cost06'] = df_merged.apply(lambda r: calc_cost(r, 'prediction'), axis=1)
    df_merged['cost4'] = df_merged.apply(lambda r: calc_cost(r, 'prediction_large'), axis=1)
    
    # Labeling Rule: Chỉ chuyển sang 4B khi cost 4B thực sự NHỎ HƠN cost 0.6B
    df_merged['route'] = (df_merged['cost4'] < df_merged['cost06']).astype(int)
    
    # Đổi tên prediction của 0.6B thành prediction_small cho rõ ràng
    df_merged = df_merged.rename(columns={'prediction': 'prediction_small'})
    
    # In một số phân bổ cơ bản
    print(f"📊 Phân bổ Route trên {split_name}:")
    print(df_merged['route'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
    
    print(f"💰 Trung bình Cost 0.6B: {df_merged['cost06'].mean():.3f}")
    print(f"💰 Trung bình Cost 4B:   {df_merged['cost4'].mean():.3f}")
    
    # Lưu ra file mới
    df_merged.to_parquet(out_path, index=False)
    print(f"✅ Đã lưu Router Dataset: {out_path.name}")

def main():
    print("🚀 Bắt đầu Build Router Dataset")
    
    # Load cost matrix
    if not os.path.exists(COST_MATRIX_PATH):
        print(f"❌ Không tìm thấy Cost Matrix tại {COST_MATRIX_PATH}")
        return
        
    with open(COST_MATRIX_PATH, "r", encoding="utf-8") as f:
        cost_matrix = yaml.safe_load(f)
        
    print(f"✅ Đã load Cost Matrix v1")
    
    # Áp dụng cho cả Train và Test
    build_dataset_for_split("train", cost_matrix)
    build_dataset_for_split("test", cost_matrix)

if __name__ == "__main__":
    main()
