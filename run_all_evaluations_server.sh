#!/bin/bash

echo "🚀 BẮT ĐẦU CHẠY ĐÁNH GIÁ TRÊN SERVER GPU 🚀"
echo "================================================="

echo "1. ĐÁNH GIÁ BASELINE (0.6B VÀ 4B)"
echo "-------------------------------------------------"
echo "Chạy 0.6B Baseline cho tập V3 (Bổ sung):"
python scripts/evaluation/evaluate_qwen3guard_lora.py --base_model "Qwen/Qwen3Guard-Gen-0.6B" --test_file "data/splits/test_final_original_v3.jsonl" --output_file "outputs/evaluation/eval_baseline_internal_v3_06b.jsonl"

echo ""
echo "Chạy 4B Baseline (Có thêm cờ --prefill_safety để chặn <think>):"
python scripts/evaluation/evaluate_qwen3guard_lora.py --base_model "Qwen/Qwen3Guard-Gen-4B" --test_file "data/splits/jbb_behaviors_benchmark.jsonl" --output_file "outputs/evaluation/eval_baseline_jbb_behaviors_4b.jsonl" --load_in_4bit --prefill_safety

python scripts/evaluation/evaluate_qwen3guard_lora.py --base_model "Qwen/Qwen3Guard-Gen-4B" --test_file "data/splits/prompt_injections_benchmark.jsonl" --output_file "outputs/evaluation/eval_baseline_prompt_injections_4b.jsonl" --load_in_4bit --prefill_safety

python scripts/evaluation/evaluate_qwen3guard_lora.py --base_model "Qwen/Qwen3Guard-Gen-4B" --test_file "data/splits/test_final_original_v3.jsonl" --output_file "outputs/evaluation/eval_baseline_internal_v3_4b.jsonl" --load_in_4bit --prefill_safety

# python scripts/evaluation/evaluate_qwen3guard_lora.py --base_model "Qwen/Qwen3Guard-Gen-4B" --test_file "data/splits/wildjailbreak_benchmark.jsonl" --output_file "outputs/evaluation/eval_baseline_wildjailbreak_4b.jsonl" --load_in_4bit --prefill_safety


echo -e "\n2. ĐÁNH GIÁ ROUTER E2E (Chạy lại để đo Latency mới)"
echo "-------------------------------------------------"
python scripts/evaluation/evaluate_benchmark_e2e.py --dataset "data/splits/jbb_behaviors_benchmark.jsonl"
python scripts/evaluation/evaluate_benchmark_e2e.py --dataset "data/splits/prompt_injections_benchmark.jsonl"
python scripts/evaluation/evaluate_benchmark_e2e.py --dataset "data/splits/test_final_original_v3.jsonl"
# python scripts/evaluation/evaluate_benchmark_e2e.py --dataset "data/splits/wildjailbreak_benchmark.jsonl"


# echo -e "\n3. ĐÁNH GIÁ BÙ TẬP WILDJAILBREAK BỊ THIẾU (4B LoRA)"
# echo "-------------------------------------------------"
# python scripts/evaluation/evaluate_qwen3guard_lora.py --base_model "Qwen/Qwen3Guard-Gen-4B" --adapter_path "outputs/qwen3guard_4b_lora_v3" --test_file "data/splits/wildjailbreak_benchmark.jsonl" --output_file "outputs/evaluation/eval_wildjailbreak_4b_lora.jsonl" --load_in_4bit


echo -e "\n4. TỔNG HỢP TOÀN BỘ BÁO CÁO (So sánh 0.6B, 4B, Router, Baseline)"
echo "-------------------------------------------------"
python scripts/evaluation/compare_models.py --file_06b "outputs/evaluation/eval_jbb_behaviors_06b.jsonl" --file_4b "outputs/evaluation/eval_jbb_behaviors_4b_lora.jsonl" --file_router "outputs/evaluation/results_jbb_behaviors_benchmark.jsonl" --file_baseline_06b "outputs/evaluation/eval_baseline_jbb_behaviors_06b.jsonl" --file_baseline_4b "outputs/evaluation/eval_baseline_jbb_behaviors_4b.jsonl" --output_txt "outputs/evaluation/report_compare_jbb_behaviors.txt"

python scripts/evaluation/compare_models.py --file_06b "outputs/evaluation/eval_prompt_injections_06b.jsonl" --file_4b "outputs/evaluation/eval_prompt_injections_4b_lora.jsonl" --file_router "outputs/evaluation/results_prompt_injections_benchmark.jsonl" --file_baseline_06b "outputs/evaluation/eval_baseline_prompt_injections_06b.jsonl" --file_baseline_4b "outputs/evaluation/eval_baseline_prompt_injections_4b.jsonl" --output_txt "outputs/evaluation/report_compare_prompt_injections.txt"

python scripts/evaluation/compare_models.py --file_06b "outputs/evaluation/eval_results_06b_lora.jsonl" --file_4b "outputs/evaluation/eval_results_4b_lora.jsonl" --file_router "outputs/evaluation/results_test_final_clean_v3.jsonl" --file_baseline_06b "outputs/evaluation/eval_baseline_internal_v3_06b.jsonl" --file_baseline_4b "outputs/evaluation/eval_baseline_internal_v3_4b.jsonl" --output_txt "outputs/evaluation/report_compare_v3.txt"

# python scripts/evaluation/compare_models.py --file_06b "outputs/evaluation/eval_wildjailbreak_06b.jsonl" --file_4b "outputs/evaluation/eval_wildjailbreak_4b_lora.jsonl" --file_router "outputs/evaluation/results_wildjailbreak_benchmark.jsonl" --file_baseline_06b "outputs/evaluation/eval_baseline_wildjailbreak_06b.jsonl" --file_baseline_4b "outputs/evaluation/eval_baseline_wildjailbreak_4b.jsonl" --output_txt "outputs/evaluation/report_compare_wildjailbreak.txt"

echo -e "\n✅ HOÀN TẤT TẤT CẢ QUÁ TRÌNH!"
echo "Bạn có thể nén toàn bộ thư mục outputs/evaluation và tải về bằng lệnh:"
echo "zip -r final_evaluation_results.zip outputs/evaluation/"
