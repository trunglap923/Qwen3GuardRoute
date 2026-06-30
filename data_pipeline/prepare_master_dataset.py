import json
import os
from collections import Counter

def main():
    input_file = "data/pipeline_merged/final_train_candidates.jsonl"
    output_file = "data/pipeline_merged/original_pool_master.jsonl"
    report_file = "data/pipeline_merged/reports/original_master_report.md"

    os.makedirs("data/pipeline_merged/reports", exist_ok=True)

    samples = []
    
    label_counter = Counter()
    lang_counter = Counter()
    group_counter = Counter()
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            # Promote final_label
            final_label = item.get("label", "unknown")
            if "evaluation" in item:
                if "final_label_v3" in item["evaluation"]:
                    final_label = item["evaluation"]["final_label_v3"]
                elif "final_label" in item["evaluation"]:
                    final_label = item["evaluation"]["final_label"]
            
            item["final_label"] = final_label
            
            # Anti-leakage schema
            item["source"] = "original"
            item["parent_sample_id"] = None
            item["augmentation_family"] = None
            item["split_version"] = "v1"
            
            samples.append(item)
            
            label_counter[final_label] += 1
            lang_counter[item.get("language", "unknown")] += 1
            group_counter[item.get("group", "unknown")] += 1

    with open(output_file, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    # Write report
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Phase 0: Original Master Dataset Report\n\n")
        f.write(f"**Total Samples:** {len(samples)}\n\n")
        
        f.write("## Label Distribution\n")
        for k, v in label_counter.most_common():
            f.write(f"- {k}: {v} ({v/len(samples)*100:.1f}%)\n")
            
        f.write("\n## Language Distribution\n")
        for k, v in lang_counter.most_common():
            f.write(f"- {k}: {v} ({v/len(samples)*100:.1f}%)\n")
            
        f.write("\n## Group Distribution (Top 15)\n")
        for k, v in group_counter.most_common(15):
            f.write(f"- {k}: {v}\n")
            
    print(f"Processed {len(samples)} samples.")
    print(f"Output saved to {output_file}")
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    main()
