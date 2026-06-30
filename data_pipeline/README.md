# Qwen3Guard Data Pipeline

This directory contains the core scripts for building the training data for the **Qwen3Guard (Router & LLM Guardrails)** system, covering everything from the initial prompt planning (Combination Plan) to the final train/val/test data splits.

## 🚀 Pipeline Architecture

The data processing workflow is divided into 4 main stages, executed sequentially as follows:

### Stage 1: Generation
1. **`generate_jobs.py`**:
   - Reads the `data/combination_plan.json` (or `data/combination_test_plan.json`) configuration file, which contains the generation formulas, domain distributions, language ratios, and expected message lengths.
   - Automatically generates role-playing prompts (system prompts) to call Large Language Model (LLM) APIs.
   - **Output:** `data/generation_jobs.jsonl`

2. **`run_generate_pilot.py`**:
   - Executes batch API calls (OpenAI/Anthropic) based on the tasks created in step 1 to simulate realistic attack/defense conversations.
   - **Output:** Raw generated data `generated_raw.jsonl`

### Stage 2: Judging & Relabeling
3. **`judge_and_relabel.py`**:
   - Utilizes an LLM-as-a-Judge (Judge 1) to review the generated conversations and classify the Semantic Intent label: `Safe`, `Unsafe`, or `Controversial`.

4. **`run_judge2_audit.py`**:
   - Uses a second independent LLM-as-a-Judge (Judge 2) to perform cross-checking.
   - This is particularly useful for handling ambiguous cases (Controversial/Hard samples), ensuring the highest quality of ground-truth labels.

### Stage 3: Consolidation
5. **`build_final_train_candidates.py`**:
   - Extracts data samples that passed the moderation phase and have confirmed quality (e.g., `keep_verified`, `relabel_confirmed`, `auto_resolved`).
   - Filters out junk and corrupted samples.
   - **Output:** `final_train_candidates.jsonl`

6. **`prepare_master_dataset.py`**:
   - Reformats the data to a standardized schema and adds anti-leakage metadata.
   - Generates statistical distribution reports of the original dataset.

7. **`merge_master.py`**:
   - Merges the Original Data with any synthetic Augmentation datasets. Performs random shuffling.
   - **Output:** The master dataset `master_train_vX.jsonl`.

### Stage 4: Splitting & Verification
8. **`split_master_for_training.py`**:
   - Uses `scikit-learn` to split the `master_train` dataset into `ft_train` (SFT), `ft_val` (Validation), and `router_pool` (used to train the XGBoost/Router models).
   - Ensures stratified splitting to maintain identical label distributions across all splits.

9. **`verify_before_train.py`**:
   - The final safety net! Scans all Train, Val, Test, and Router splits to ensure that **no single `sample_id` overlaps across splits** (Zero Data Leakage).

---

## 💻 Quick Start Guide

You can run the entire pipeline sequentially from top to bottom. The scripts are pre-configured with relative paths to fetch data from the `data/` directory.

```bash
# Stage 1
python data_pipeline/generate_jobs.py
python data_pipeline/run_generate_pilot.py

# Stage 2
python data_pipeline/judge_and_relabel.py
python data_pipeline/run_judge2_audit.py

# Stage 3
python data_pipeline/build_final_train_candidates.py
python data_pipeline/prepare_master_dataset.py
python data_pipeline/merge_master.py

# Stage 4
python data_pipeline/split_master_for_training.py
python data_pipeline/verify_before_train.py
```
