"""
verify_before_train.py
Chạy 5 kiểm tra trước khi train:
1. sample_id trùng lặp trong từng file
2. Overlap sample_id giữa train/val/router/test splits
3. SFT train/val có đúng format 2 messages không
4. assistant output có đúng label format không
5. YAML trỏ đúng file không
"""

import json
import sys
from pathlib import Path
from collections import Counter

# ─── Paths ────────────────────────────────────────────────────────────────────
SPLITS = {
    "ft_train":    Path("data/splits/ft_train_v1.jsonl"),
    "ft_val":      Path("data/splits/ft_val_v1.jsonl"),
    "router_pool": Path("data/splits/router_pool_v1.jsonl"),
    "test_final":  Path("data/splits/test_final_original_v1.jsonl"),
}

SFT_FILES = {
    "sft_train": Path("data/training/qwen3guard/train_sft_qwen3guard_v1.jsonl"),
    "sft_val":   Path("data/training/qwen3guard/val_sft_qwen3guard_v1.jsonl"),
}

YAML_PATH = Path("configs/qwen3guard_06b_lora_v1.yaml")

VALID_LABELS = {"Safety: Safe", "Safety: Controversial", "Safety: Unsafe"}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_jsonl(path: Path):
    items = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ❌ JSON parse error at line {i}: {e}")
    return items

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

all_passed = True

# ─── Check 1: sample_id duplicates trong mỗi split ───────────────────────────
print("\n" + "="*70)
print("CHECK 1: sample_id trùng lặp trong từng split file")
print("="*70)

split_id_sets = {}  # tên split -> set sample_id

for name, path in SPLITS.items():
    if not path.exists():
        print(f"  {FAIL} File không tồn tại: {path}")
        all_passed = False
        split_id_sets[name] = set()
        continue
    items = load_jsonl(path)
    ids = [x.get("sample_id") for x in items]
    counter = Counter(ids)
    dupes = {k: v for k, v in counter.items() if v > 1}
    split_id_sets[name] = set(ids)
    if dupes:
        print(f"  {FAIL} [{name}] {len(dupes)} sample_id bị trùng: {list(dupes.items())[:5]}")
        all_passed = False
    else:
        print(f"  {PASS} [{name}] {len(ids)} samples — không có trùng lặp")

# ─── Check 2: Overlap giữa các splits ────────────────────────────────────────
print("\n" + "="*70)
print("CHECK 2: Overlap sample_id giữa các splits")
print("="*70)

split_names = list(split_id_sets.keys())
for i in range(len(split_names)):
    for j in range(i + 1, len(split_names)):
        a, b = split_names[i], split_names[j]
        overlap = split_id_sets[a] & split_id_sets[b]
        if overlap:
            print(f"  {FAIL} [{a}] ∩ [{b}] = {len(overlap)} mẫu bị chồng! Examples: {list(overlap)[:3]}")
            all_passed = False
        else:
            print(f"  {PASS} [{a}] ∩ [{b}] = 0 — không overlap")

# ─── Check 3 & 4: SFT format + label format ──────────────────────────────────
print("\n" + "="*70)
print("CHECK 3 & 4: SFT format (2 messages) và assistant label hợp lệ")
print("="*70)

for name, path in SFT_FILES.items():
    if not path.exists():
        print(f"  {FAIL} File không tồn tại: {path}")
        all_passed = False
        continue

    items = load_jsonl(path)
    fmt_errors = []
    label_errors = []

    for idx, item in enumerate(items):
        msgs = item.get("messages", [])
        # Check format: phải có đúng 2 messages [user, assistant]
        if len(msgs) != 2:
            fmt_errors.append((idx, f"có {len(msgs)} messages"))
            continue
        if msgs[0].get("role") != "user":
            fmt_errors.append((idx, f"role[0]={msgs[0].get('role')}"))
            continue
        if msgs[1].get("role") != "assistant":
            fmt_errors.append((idx, f"role[1]={msgs[1].get('role')}"))
            continue

        # Check label: "Safety: Safe/Controversial/Unsafe"
        output = msgs[1].get("content", "").strip()
        if output not in VALID_LABELS:
            label_errors.append((idx, repr(output)))

    if fmt_errors:
        print(f"  {FAIL} [{name}] {len(fmt_errors)} lỗi format. Ví dụ: {fmt_errors[:3]}")
        all_passed = False
    else:
        print(f"  {PASS} [{name}] {len(items)} samples — format 2 messages đúng")

    if label_errors:
        print(f"  {FAIL} [{name}] {len(label_errors)} label sai. Ví dụ: {label_errors[:3]}")
        all_passed = False
    else:
        print(f"  {PASS} [{name}] tất cả assistant output đúng format 'Safety: X'")

# ─── Check 5: YAML trỏ đúng file ─────────────────────────────────────────────
print("\n" + "="*70)
print("CHECK 5: YAML train_file / eval_file trỏ đúng")
print("="*70)

if not YAML_PATH.exists():
    print(f"  {FAIL} YAML không tồn tại: {YAML_PATH}")
    all_passed = False
else:
    import yaml
    with open(YAML_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for field in ("train_file", "eval_file"):
        val = cfg.get(field)
        if val is None:
            print(f"  {FAIL} YAML thiếu field: {field}")
            all_passed = False
        elif not Path(val).exists():
            print(f"  {FAIL} [{field}] = '{val}' — FILE KHÔNG TỒN TẠI")
            all_passed = False
        else:
            print(f"  {PASS} [{field}] = '{val}' — tồn tại")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
if all_passed:
    print("🎉 TẤT CẢ CHECKS PASS — sẵn sàng backup và chạy dry-run!")
else:
    print("🚨 CÓ LỖI — vui lòng sửa trước khi chạy train!")
print("="*70 + "\n")

sys.exit(0 if all_passed else 1)
