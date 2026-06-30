import json
import random
import argparse
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent.parent
AUG_FILE = BASE_DIR / "data" / "pipeline_merged" / "augmentation_after_leakage_v1.jsonl"
ORIGINAL_TRAIN_FILE = BASE_DIR / "data" / "pipeline_merged" / "final_train_candidates.jsonl"

OUT_MASTER_FILE = BASE_DIR / "data" / "pipeline_merged" / "master_train_v1.jsonl"
REPORT_FILE = BASE_DIR / "data" / "pipeline_merged" / "reports" / "master_merge_report_v1.json"

RANDOM_SEED = 42

def main():
    if not AUG_FILE.exists():
        print(f"❌ Không tìm thấy {AUG_FILE}")
        return
    if not ORIGINAL_TRAIN_FILE.exists():
        print(f"❌ Không tìm thấy {ORIGINAL_TRAIN_FILE}")
        return
        
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "original_train_count": 0,
        "augmentation_count": 0,
        "master_train_count": 0,
        "label_distribution_before_merge_original": Counter(),
        "label_distribution_augmentation": Counter(),
        "label_distribution_master": Counter(),
        "combo_distribution_augmentation": Counter(),
        "random_seed": RANDOM_SEED,
        "inputs": [str(ORIGINAL_TRAIN_FILE.name), str(AUG_FILE.name)],
        "output": str(OUT_MASTER_FILE.name)
    }
    
    master_samples = []
    
    # 1. Load Original Train
    with open(ORIGINAL_TRAIN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            s = json.loads(line)
            master_samples.append(s)
            stats["original_train_count"] += 1
            lbl = s.get("label")
            stats["label_distribution_before_merge_original"][lbl] += 1
            stats["label_distribution_master"][lbl] += 1
            
    print(f"📥 Đã tải {stats['original_train_count']} mẫu từ Original Train.")
    
    # 2. Load Augmentation
    aug_samples = []
    with open(AUG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            s = json.loads(line)
            aug_samples.append(s)
            stats["augmentation_count"] += 1
            lbl = s.get("label")
            stats["label_distribution_augmentation"][lbl] += 1
            stats["label_distribution_master"][lbl] += 1
            stats["combo_distribution_augmentation"][s.get("combo_id", "")] += 1
            
    print(f"📥 Đã tải {stats['augmentation_count']} mẫu từ Augmentation.")
    
    # 3. Merge and Shuffle
    master_samples.extend(aug_samples)
    stats["master_train_count"] = len(master_samples)
    
    random.seed(RANDOM_SEED)
    random.shuffle(master_samples)
    
    # 4. Export
    with open(OUT_MASTER_FILE, 'w', encoding='utf-8') as f:
        for s in master_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Quá trình Merge hoàn tất!")
    print(f"Báo cáo: {REPORT_FILE.name}")
    print(f"Output Master Train: {OUT_MASTER_FILE.name} ({stats['master_train_count']} mẫu)")
    
    print("\n📊 Báo cáo Merge:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
