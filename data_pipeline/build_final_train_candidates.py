import json

INPUT = 'data/original_judged_v3_3_resolved.jsonl'
OUTPUT = 'data/final_train_candidates.jsonl'

def build_candidates():
    candidates = []
    skipped_manual = 0
    unsafe_auto = 0
    with open(INPUT, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            s = json.loads(line)
            ev = s.get('evaluation', {})
            status = ev.get('clean_status_final')
            status_v3 = ev.get('clean_status_v3')
            
            # Use immediately for training:
            if status in ["keep_verified", "relabel_confirmed", "keep_unverified"]:
                candidates.append(s)
            elif status_v3 == "auto_resolved":
                # We include all auto_resolved (controversial + unsafe).
                # The user will have manually edited the 20 unsafe ones in the JSONL if they wanted to change them.
                # Or if they just want to exclude them pending review, they can do it.
                # But the instructions said: "Tạo file final_train_candidates.jsonl từ: keep_verified + relabel_confirmed + keep_unverified + auto_resolved đã kiểm tra."
                if ev.get("final_label_v3") == "unsafe":
                    unsafe_auto += 1
                candidates.append(s)
            elif status_v3 == "manual_review":
                skipped_manual += 1

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
            
    print(f"Built final training candidates: {OUTPUT}")
    print(f"Total samples included: {len(candidates)}")
    print(f"  - Including {unsafe_auto} auto-resolved unsafe samples (ensure these were reviewed).")
    print(f"Total remaining manual_review excluded: {skipped_manual}")

if __name__ == "__main__":
    build_candidates()
