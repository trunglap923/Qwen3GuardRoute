# Qwen3Guard Router System 🛡️

Qwen3Guard is a high-performance content moderation (Guardrails) system for Large Language Model (LLM) applications, built on the Qwen2.5 architecture. The system combines deeply fine-tuned LLMs (0.6B and 4B using LoRA) with a lightweight **Router model (XGBoost/LightGBM)** to optimize the trade-offs between **Accuracy**, **Unsafe Recall**, and **Latency**.

## 🌟 Key Features

- **Multi-class Safety Guardrail:** Instead of relying on a binary assessment (Safe/Unsafe), Qwen3Guard recognizes a **Controversial** label (borderline/ambiguous cases). This makes the system much more flexible when handling academic, hypothetical, or security research queries.
- **Ultra-fast End-to-End Router:** The Router leverages Logit Evaluation (extracting probabilities from a single forward-pass of the 0.6B model) to determine whether to escalate to the 4B model. The average latency is approximately **~0.18s**, which is even faster than running the 0.6B model standalone in generation mode.
- **Complete Data Pipeline:** Includes an automated generation, cross-evaluation (LLM-as-a-Judge), and professional data consolidation pipeline to create high-quality SFT training datasets that are completely immune to data leakage (Zero Data Leakage).

## 📂 Repository Structure

```text
qwen3guard_github_export/
├── data_pipeline/      # Scripts for automated generation, evaluation, and data splitting
├── scripts/
│   ├── training/       # Pipeline to train the LLM Guards and the Router (XGBoost)
│   └── evaluation/     # Scripts for benchmarking, latency measurement, and model comparison
├── configs/            # YAML files for Router configs, LoRA hyperparameters, and Cost Matrix
├── BENCHMARK_GUIDE.md  # Detailed instructions on how to evaluate across different benchmarks
├── run_all_evaluations_server.sh  # Automated script for running end-to-end evaluations
├── train.sh            # Script to trigger SFT Training
├── eval.sh             # Script to trigger Evaluation
└── setup_vastai.sh     # Script to initialize the environment on a GPU cloud server (e.g., Vast.ai)
```

## 🚀 Usage Guide

### 1. Environment Setup
If you are running on a cloud GPU server like Vast.ai, grant execution permissions and run the setup script:
```bash
chmod +x setup_vastai.sh
./setup_vastai.sh
```

### 2. Building the Dataset (Data Pipeline)
All source code to generate, evaluate, and consolidate the training dataset is located in the `data_pipeline/` directory. 
👉 Read the [detailed Data Pipeline guide here](data_pipeline/README.md).

### 3. Model Training
Run the `train.sh` script to trigger Fine-tuning (LoRA) for the language models (0.6B/4B).
To train the Router model (balancing Cost-Latency), run:
```bash
python scripts/training/train_router.py
```

### 4. Evaluation & Benchmarking
You can evaluate standalone models (0.6B, 4B) or the E2E Router system on various Benchmarks such as *JBB-Behaviors*, *Prompt Injections*, and *WildJailbreak*.

Run the Bash script to execute the full comparative workflow:
```bash
./run_all_evaluations_server.sh
```
👉 Please refer to `BENCHMARK_GUIDE.md` for a deeper understanding of the evaluation methodology and metrics (Accuracy, Macro F1, Unsafe Recall, Decision Cost).

## 📊 Performance Results (E2E Router)

Thanks to the **Logit Evaluation** technique, the Router system bypasses auto-regressive decoding, yielding an incredibly impressive response time while maintaining an Unsafe Recall comparable to massive parameter models.

| System          | Accuracy   | Macro F1   | Unsafe Recall   | Avg Cost   | Latency(s) |
|-----------------|------------|------------|-----------------|------------|------------|
| Baseline 0.6B   | 0.6200     | 0.4511     | 0.8190          | 0.8867     | 0.2179     |
| Baseline 4B     | 0.6633     | 0.4588     | 0.9655          | 0.8667     | 0.5059     |
| 0.6B LoRA       | 0.6867     | 0.4709     | 0.8448          | 0.9133     | 0.3689     |
| 4B LoRA         | 0.6367     | 0.4481     | 0.9310          | 0.8933     | 0.4947     |
| **Router E2E**  | **0.6767** | **0.4634** | **0.8966**      | **0.9300** | **0.1813** |

*Note: Latency was measured empirically on NVIDIA RTX hardware.*
