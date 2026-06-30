import os
import json
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import warnings
import time
# Ignore some scikit-learn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

from router_inference import SafeRouteSystem

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "router_config_v1.yaml"
ROUTER_MODEL_PATH = BASE_DIR / "outputs" / "router" / "best_router_model.joblib"
COST_MATRIX_PATH = BASE_DIR / "configs" / "cost_matrix_v1.yaml"

def calculate_metrics(y_true, y_pred, costs, name=""):
    valid_labels = ["safe", "controversial", "unsafe"]
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=valid_labels, average="macro", zero_division=0)
    
    unsafe_total = sum(1 for yt in y_true if yt == "unsafe")
    unsafe_correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "unsafe" and yp == "unsafe")
    unsafe_recall = unsafe_correct / unsafe_total if unsafe_total > 0 else 0
    
    avg_cost = sum(costs) / len(costs) if costs else 0
    
    return {
        "Name": name,
        "Accuracy": acc,
        "Macro F1": macro_f1,
        "Unsafe Recall": unsafe_recall,
        "Decision Cost": avg_cost,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate SafeRoute E2E on a specific benchmark")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the JSONL benchmark file (e.g., data/splits/jbb_behaviors_benchmark.jsonl)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to evaluate")
    args = parser.parse_args()

    dataset_path = BASE_DIR / args.dataset
    if not os.path.exists(dataset_path):
        print(f"❌ Không tìm thấy dataset tại {dataset_path}")
        return

    print("🚀 Bắt đầu Khởi tạo Hệ thống...")
    try:
        system = SafeRouteSystem(CONFIG_PATH, ROUTER_MODEL_PATH)
    except Exception as e:
        print(f"❌ Không thể khởi tạo SafeRouteSystem: {e}")
        return

    # Load Cost Matrix
    try:
        with open(COST_MATRIX_PATH, "r", encoding="utf-8") as f:
            cost_matrix = yaml.safe_load(f)
        print("✅ Đã load Cost Matrix v1")
    except Exception as e:
        print(f"❌ Lỗi load Cost Matrix: {e}")
        return

    # Prepare Output File
    dataset_name = dataset_path.stem
    output_dir = BASE_DIR / "outputs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"results_{dataset_name}.jsonl"

    print(f"📚 Đang tải dataset: {dataset_name}...")
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    if args.limit:
        data = data[:args.limit]
        print(f"⚠️ Chạy chế độ giới hạn: {args.limit} mẫu")

    y_true = []
    y_pred = []
    costs = []
    latencies = []
    routed_4b_count = 0

    print(f"🔄 Bắt đầu chạy Inference trên {len(data)} mẫu (Kết quả lưu tại {results_file.name})...")
    
    with open(results_file, "w", encoding="utf-8") as out_f:
        for item in tqdm(data, desc="Evaluating"):
            messages = item.get("messages", [])
            gold_label = item.get("label", "safe").lower()
            
            # Predict
            start_time = time.time()
            try:
                res = system.infer(messages)
            except Exception as e:
                print(f"Lỗi inference mẫu {item.get('sample_id', 'unknown')}: {e}")
                continue
                
            latency = time.time() - start_time
                
            pred_label = res["final_prediction"]
            routed_to = res["routed_to"]
            confidence = res["confidence"]
            
            # Tính Cost
            try:
                cost = cost_matrix[gold_label][pred_label]
            except KeyError:
                cost = 0 # Fallback
                
            y_true.append(gold_label)
            y_pred.append(pred_label)
            costs.append(cost)
            latencies.append(latency)
            
            if routed_to == "4B":
                routed_4b_count += 1
                
            # Lưu chi tiết
            out_item = {
                "sample_id": item.get("sample_id", ""),
                "source": item.get("source", ""),
                "gold_label": gold_label,
                "prediction": pred_label,
                "routed_to": routed_to,
                "confidence": confidence,
                "cost": cost,
                "latency_sec": latency
            }
            out_f.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            
    print(f"✅ Đã lưu kết quả chi tiết từng câu hỏi tại: {results_file}")
    
    if not y_true:
        print("❌ Không có dữ liệu để đánh giá.")
        return
        
    # Tính toán Metric
    metrics = calculate_metrics(y_true, y_pred, costs, name="SafeRoute(Ours)")
    metrics["Latency (s)"] = sum(latencies) / len(latencies) if latencies else 0.0
    usage_4b = (routed_4b_count / len(y_true)) * 100
    
    print("\n" + "="*95)
    print(f"🏆 KẾT QUẢ ĐÁNH GIÁ TRÊN BỘ: {dataset_name.upper()} 🏆")
    print("="*95)
    header = f"| {'System':<18} | {'Accuracy':<8} | {'Macro F1':<8} | {'Unsafe Recall':<13} | {'Dec. Cost':<9} | {'Latency':<8} | {'Large Usage':<11} |"
    print(header)
    print("|" + "-"*20 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*15 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*13 + "|")
    
    acc = f"{metrics['Accuracy']:.4f}"
    f1 = f"{metrics['Macro F1']:.4f}"
    rec = f"{metrics['Unsafe Recall']:.4f}"
    cost = f"{metrics['Decision Cost']:.3f}"
    lat = f"{metrics['Latency (s)']:.3f}s"
    use = f"{usage_4b:.1f}%"
    print(f"| {metrics['Name']:<18} | {acc:<8} | {f1:<8} | {rec:<13} | {cost:<9} | {lat:<8} | {use:<11} |")
    print("="*95)
    
    print("\n📊 CHI TIẾT TỪNG NHÃN (CLASSIFICATION REPORT)")
    valid_labels = ["safe", "controversial", "unsafe"]
    print(classification_report(y_true, y_pred, labels=valid_labels, digits=4, zero_division=0))
    
    print("Confusion Matrix (Rows=Gold, Cols=Pred):")
    print(valid_labels)
    print(confusion_matrix(y_true, y_pred, labels=valid_labels))
    
    summary_file = output_dir / f"summary_{dataset_name}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(header + "\n")
        f.write("|" + "-"*20 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*15 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*13 + "|\n")
        f.write(f"| {metrics['Name']:<18} | {acc:<8} | {f1:<8} | {rec:<13} | {cost:<9} | {lat:<8} | {use:<11} |\n\n")
        f.write(classification_report(y_true, y_pred, labels=valid_labels, digits=4, zero_division=0))
        
    print(f"✅ Đã lưu tóm tắt báo cáo tại: {summary_file}")

if __name__ == "__main__":
    main()
