import json
import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================

EVAL_FILE = "eval_06b_ft_test_final.jsonl"
TEST_FILE = "test_final_original_v1.jsonl"

OUT_DIR = Path("error_analysis")
OUT_DIR.mkdir(exist_ok=True)

# =========================
# LOAD
# =========================

eval_rows = []

with open(EVAL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        eval_rows.append(json.loads(line))

test_rows = []

with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        test_rows.append(json.loads(line))

eval_df = pd.DataFrame(eval_rows)
test_df = pd.DataFrame(test_rows)

# =========================
# JOIN
# =========================

meta_cols = [
    "sample_id",
    "combo_id",
    "attack_type",
    "scenario",
    "mechanism",
    "evasion",
    "language",
    "group",
    "domain",
    "style",
    "n_messages"
]

df = eval_df.merge(
    test_df[meta_cols],
    on="sample_id",
    how="left",
    suffixes=("", "_meta")
)

# =========================
# NORMALIZE
# =========================

df["gold_label"] = df["gold_label"].str.lower()
df["pred_label"] = df["pred_label"].str.lower()

df["is_error"] = (
    df["gold_label"] != df["pred_label"]
)

# =========================
# OVERALL
# =========================

overall = {
    "total": len(df),
    "correct": (~df["is_error"]).sum(),
    "errors": df["is_error"].sum(),
    "accuracy": (~df["is_error"]).mean()
}

pd.DataFrame([overall]).to_csv(
    OUT_DIR / "overall.csv",
    index=False
)

# =========================
# GROUP ANALYSIS
# =========================

GROUP_COLS = [
    "attack_type",
    "scenario",
    "mechanism",
    "evasion",
    "language",
    "group",
    "combo_id",
    "n_messages"
]

for col in GROUP_COLS:

    stats = (
        df.groupby(col)
        .agg(
            total=("sample_id", "count"),
            errors=("is_error", "sum")
        )
        .reset_index()
    )

    stats["error_rate"] = (
        stats["errors"] /
        stats["total"]
    )

    stats = stats.sort_values(
        "error_rate",
        ascending=False
    )

    stats.to_csv(
        OUT_DIR / f"error_by_{col}.csv",
        index=False
    )

# =========================
# CONFUSION SUBSETS
# =========================

unsafe_to_safe = df[
    (df["gold_label"] == "unsafe")
    &
    (df["pred_label"] == "safe")
]

safe_to_unsafe = df[
    (df["gold_label"] == "safe")
    &
    (df["pred_label"] == "unsafe")
]

controversial_to_safe = df[
    (df["gold_label"] == "controversial")
    &
    (df["pred_label"] == "safe")
]

unsafe_to_safe.to_csv(
    OUT_DIR / "unsafe_to_safe.csv",
    index=False
)

safe_to_unsafe.to_csv(
    OUT_DIR / "safe_to_unsafe.csv",
    index=False
)

controversial_to_safe.to_csv(
    OUT_DIR / "controversial_to_safe.csv",
    index=False
)

# =========================
# HARDEST COMBOS
# =========================

combo_stats = (
    df.groupby("combo_id")
    .agg(
        total=("sample_id", "count"),
        errors=("is_error", "sum")
    )
    .reset_index()
)

combo_stats["error_rate"] = (
    combo_stats["errors"]
    / combo_stats["total"]
)

combo_stats = combo_stats[
    combo_stats["total"] >= 5
]

combo_stats = combo_stats.sort_values(
    "error_rate",
    ascending=False
)

combo_stats.to_csv(
    OUT_DIR / "hardest_combos.csv",
    index=False
)

# =========================
# TOP ERROR SAMPLES
# =========================

error_samples = df[
    df["is_error"]
]

error_samples.to_csv(
    OUT_DIR / "all_error_samples.csv",
    index=False
)

print("Done.")
print("Output:", OUT_DIR)