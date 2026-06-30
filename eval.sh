#!/bin/bash
echo "📊 Bắt đầu đánh giá (Evaluate) mô hình..."

python scripts/evaluation/evaluate_qwen3guard_lora.py --config configs/qwen3guard_06b_lora_v1.yaml
python scripts/evaluation/error_analysis.py

echo "📦 Đang đóng gói kết quả để tải về..."
zip -r outputs.zip outputs/
echo "✅ Hoàn tất! Bạn có thể tải file outputs.zip về máy."
