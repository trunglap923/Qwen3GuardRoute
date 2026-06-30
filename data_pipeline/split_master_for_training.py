import json
from pathlib import Path
from collections import Counter
import hashlib
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MASTER_FILE = BASE_DIR / "data" / "pipeline_merged" / "master_train_v1.jsonl"
OUT_DIR = BASE_DIR / "data" / "pipeline_merged" / "splits"
REPORT_DIR = BASE_DIR / "data" / "pipeline_merged" / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FT_TRAIN_FILE = OUT_DIR / "ft_train_v1.jsonl"
FT_VAL_FILE = OUT_DIR / "ft_val_v1.jsonl"
ROUTER_POOL_FILE = OUT_DIR / "router_pool_v1.jsonl"
REPORT_FILE = REPORT_DIR / "split_master_report_v1.json"

def calculate_checksum(filepath):
    if not filepath.exists(): return None
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def infer_source(s):
    if s.get("source") in {"original", "augmentation"}:
        return s["source"]
    combo_id = str(s.get("combo_id", ""))
    sample_id = str(s.get("sample_id", ""))
    job_id = str(s.get("job_id", ""))
    group = str(s.get("group", ""))
    if (
        combo_id.startswith("AUG_")
        or sample_id.startswith("AUG_")
        or job_id.startswith("AUG_")
        or group.startswith("aug_")
    ):
        return "augmentation"
    return "original"

def get_stats(data_list):
    stats = {
        "total": len(data_list),
        "labels": Counter(),
        "sources": Counter(),
        "groups": Counter()
    }
    for s in data_list:
        lbl = s.get("label", "unknown")
        stats["labels"][lbl] += 1
        
        src = infer_source(s)
        stats["sources"][src] += 1
        
        grp = s.get("group", "unknown")
        stats["groups"][grp] += 1
        
    return stats

def main():
    print(f"🔄 Đang tải dữ liệu từ: {MASTER_FILE.name}")
    data = []
    strata = []
    
    with open(MASTER_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            s = json.loads(line)
            data.append(s)
            
            lbl = s.get("label", "unknown")
            src = infer_source(s)
            strata.append(f"{lbl}_{src}")

    total_samples = len(data)
    print(f"📊 Tổng số mẫu: {total_samples}")
    
    # Bước 1: Tách 20% Router Pool, 80% Temp Train
    temp_data, router_pool_data, temp_strata, _ = train_test_split(
        data, strata, test_size=0.20, random_state=42, stratify=strata
    )
    
    # Bước 2: Tách 12.5% của Temp Train làm Validation (0.125 * 0.8 = 0.10 của tổng)
    ft_train_data, ft_val_data = train_test_split(
        temp_data, test_size=0.125, random_state=42, stratify=temp_strata
    )
    
    # Kiểm tra ID overlap
    train_ids = set(s["sample_id"] for s in ft_train_data)
    val_ids = set(s["sample_id"] for s in ft_val_data)
    router_ids = set(s["sample_id"] for s in router_pool_data)
    
    assert len(train_ids.intersection(val_ids)) == 0, "Lỗi rò rỉ: Train và Val trùng ID!"
    assert len(train_ids.intersection(router_ids)) == 0, "Lỗi rò rỉ: Train và Router trùng ID!"
    assert len(val_ids.intersection(router_ids)) == 0, "Lỗi rò rỉ: Val và Router trùng ID!"
    
    # Ghi file
    def write_jsonl(filepath, data_list):
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    write_jsonl(FT_TRAIN_FILE, ft_train_data)
    write_jsonl(FT_VAL_FILE, ft_val_data)
    write_jsonl(ROUTER_POOL_FILE, router_pool_data)
    
    # Tạo report
    report = {
        "random_seed": 42,
        "input_checksum": calculate_checksum(MASTER_FILE),
        "total_samples_before_split": total_samples,
        "splits": {
            "ft_train_v1": {
                "filepath": str(FT_TRAIN_FILE.relative_to(BASE_DIR)),
                "checksum": calculate_checksum(FT_TRAIN_FILE),
                "stats": get_stats(ft_train_data)
            },
            "ft_val_v1": {
                "filepath": str(FT_VAL_FILE.relative_to(BASE_DIR)),
                "checksum": calculate_checksum(FT_VAL_FILE),
                "stats": get_stats(ft_val_data)
            },
            "router_pool_v1": {
                "filepath": str(ROUTER_POOL_FILE.relative_to(BASE_DIR)),
                "checksum": calculate_checksum(ROUTER_POOL_FILE),
                "stats": get_stats(router_pool_data)
            }
        }
    }
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
        
    print("✅ Đã chia tách dữ liệu hoàn tất!")
    print(f"  - ft_train: {len(ft_train_data)} mẫu")
    print(f"  - ft_val: {len(ft_val_data)} mẫu")
    print(f"  - router_pool: {len(router_pool_data)} mẫu")
    print(f"Báo cáo chi tiết lưu tại: {REPORT_FILE.name}")

if __name__ == "__main__":
    main()
