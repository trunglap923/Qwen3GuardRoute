import os
import yaml
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "router_config_v1.yaml"
SPLITS_DIR = BASE_DIR / "data" / "splits"
OUT_DIR = BASE_DIR / "outputs" / "router"

def load_data(split_name):
    path = SPLITS_DIR / f"router_dataset_v1_{split_name}.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy {path}")
    return pd.read_parquet(path)

def extract_X_y(df, feature_set):
    """
    Trích xuất ma trận X và vector y theo cấu hình feature set.
    Hiện tại pipeline đã được đơn giản hóa, chỉ sử dụng duy nhất 'hidden' làm feature.
    """
    if "hidden" not in feature_set:
        raise ValueError("Feature set must contain 'hidden'")
        
    X = np.stack(df['hidden'].values)
    y = df['route'].values
    
    return X, y

def evaluate_router(y_true, y_pred, df_test):
    """
    Tính toán các chỉ số an toàn và học máy của SafeRoute.
    """
    # 1. Standard ML Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    # y_true=1 (Nên chuyển 4B), y_pred=0 (Giữ 0.6B) -> False Negative
    # Đây là lỗi nguy hiểm: Bỏ lọt mẫu đáng lẽ phải đẩy lên 4B
    try:
        fn = cm[1][0]
        actual_positives = cm[1].sum()
        false_route_rate = fn / actual_positives if actual_positives > 0 else 0
    except IndexError:
        false_route_rate = 0
        
    # 2. SafeRoute Business Metrics
    cost06 = df_test['cost06'].values
    cost4 = df_test['cost4'].values
    
    # Cost của các hệ thống cơ sở
    avg_cost_always_06 = cost06.mean()
    avg_cost_always_4b = cost4.mean()
    
    # Oracle Cost: Luôn chọn model có cost nhỏ hơn
    oracle_cost = np.minimum(cost06, cost4).mean()
    
    # Router Cost
    # Nếu y_pred == 1 (Route) -> tốn cost4
    # Nếu y_pred == 0 (Keep) -> tốn cost06
    router_cost = np.where(y_pred == 1, cost4, cost06).mean()
    
    cost_reduction = avg_cost_always_06 - router_cost
    oracle_gap = router_cost - oracle_cost
    usage_4b = np.mean(y_pred)
    
    return {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "False Route Rate": float(false_route_rate),
        "Average Cost": float(router_cost),
        "Always 0.6B Cost": float(avg_cost_always_06),
        "Always 4B Cost": float(avg_cost_always_4b),
        "Oracle Cost": float(oracle_cost),
        "Oracle Gap": float(oracle_gap),
        "Cost Reduction": float(cost_reduction),
        "4B Usage Rate": float(usage_4b)
    }

def main():
    print("🚀 Bắt đầu quá trình Huấn luyện SafeRoute")
    
    # Đọc config
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    seed = config.get("seed", 42)
    np.random.seed(seed)
    
    feature_sets = config.get("router_settings", {}).get("feature_sets", {})
    if not feature_sets:
        feature_sets = {"A": ["hidden"]}
        
    # Load Data
    try:
        df_train = load_data("train")
        df_test = load_data("test")
    except FileNotFoundError as e:
        print(f"❌ Lỗi: {e}")
        return
        
    print(f"✅ Đã load dữ liệu: Train ({len(df_train)} mẫu), Test ({len(df_test)} mẫu)")
    
    # Định nghĩa các thuật toán Benchmark (có StandardScaler cho LR và MLP)
    models = {
        "LR": Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(random_state=seed, max_iter=1000, class_weight='balanced'))]),
        "XGB": XGBClassifier(random_state=seed, eval_metric='logloss'),
        "MLP": Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(random_state=seed, max_iter=500, hidden_layer_sizes=(128, 64)))])
    }
    
    # Số chiều PCA không được vượt quá số feature đầu vào
    try:
        df_dummy = load_data("train")
        hidden_dim = len(df_dummy['hidden'].iloc[0])
        n_samples = len(df_dummy)
    except:
        hidden_dim = 1024 # Fallback
        n_samples = 1000
        
    max_pca = min(hidden_dim, n_samples)
    n_pca_256 = min(256, max_pca)
    n_pca_128 = min(128, max_pca)
    
    # Thêm PCA Benchmarks trên Feature Set A (Dùng LR làm classifier baseline)
    pca_models = {
        "PCA_256_LR": Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=n_pca_256, random_state=seed)), ("lr", LogisticRegression(random_state=seed, max_iter=1000, class_weight='balanced'))]),
        "PCA_128_LR": Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=n_pca_128, random_state=seed)), ("lr", LogisticRegression(random_state=seed, max_iter=1000, class_weight='balanced'))])
    }
    
    results = {}
    best_model = None
    best_cost = float('inf')
    best_model_name = ""
    best_features = ""
    
    # 1. Benchmark Standard Feature Sets
    for set_name, features in feature_sets.items():
        print(f"\\n{'='*50}")
        print(f"🧪 THỬ NGHIỆM FEATURE SET {set_name}: {features}")
        
        X_train, y_train = extract_X_y(df_train, features)
        X_test, y_test = extract_X_y(df_test, features)
        
        for model_name, model in models.items():
            print(f"  ▶ Đang train {model_name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = evaluate_router(y_test, y_pred, df_test)
            run_name = f"{model_name}_Set{set_name}"
            results[run_name] = metrics
            
            print(f"    - Avg Cost: {metrics['Average Cost']:.3f} | Oracle Gap: {metrics['Oracle Gap']:.3f} | 4B Usage: {metrics['4B Usage Rate']:.1%}")
            
            # Cập nhật Best Model (Tối ưu Average Cost)
            if metrics['Average Cost'] < best_cost:
                best_cost = metrics['Average Cost']
                best_model = model
                best_model_name = run_name
                best_features = features
                
    # 2. Benchmark PCA Experiments (Chỉ dùng Feature Set A: Hidden)
    print(f"\\n{'='*50}")
    print(f"🧪 THỬ NGHIỆM DIMENSIONALITY REDUCTION (PCA)")
    if "A" in feature_sets:
        X_train_pca, y_train_pca = extract_X_y(df_train, feature_sets["A"])
        X_test_pca, y_test_pca = extract_X_y(df_test, feature_sets["A"])
        
        for model_name, model in pca_models.items():
            print(f"  ▶ Đang train {model_name}...")
            model.fit(X_train_pca, y_train_pca)
            y_pred = model.predict(X_test_pca)
            
            metrics = evaluate_router(y_test_pca, y_pred, df_test)
            results[model_name] = metrics
            
            print(f"    - Avg Cost: {metrics['Average Cost']:.3f} | Oracle Gap: {metrics['Oracle Gap']:.3f} | 4B Usage: {metrics['4B Usage Rate']:.1%}")
            
            if metrics['Average Cost'] < best_cost:
                best_cost = metrics['Average Cost']
                best_model = model
                best_model_name = model_name
                best_features = feature_sets["A"]
    else:
        print("⚠️ Bỏ qua PCA vì không tìm thấy Feature Set 'A' trong config.")
        
    # 3. Lưu kết quả
    if best_model is None:
        raise RuntimeError("No router model was trained successfully.")
        
    print(f"\\n{'='*50}")
    print(f"🏆 BEST ROUTER MODEL: {best_model_name} (Cost: {best_cost:.3f})")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Lưu file report JSON
    with open(OUT_DIR / "metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    # Lưu Joblib
    joblib_path = OUT_DIR / "best_router_model.joblib"
    metadata = {
        "model_name": best_model_name,
        "features_used": best_features,
        "metrics": results[best_model_name],
        "model_object": best_model
    }
    joblib.dump(metadata, joblib_path)
    
    print(f"✅ Đã lưu report tại: {OUT_DIR}/metrics_report.json")
    print(f"✅ Đã lưu model tại:  {joblib_path}")

if __name__ == "__main__":
    main()
