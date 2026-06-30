#!/bin/bash
echo "🚀 Bắt đầu huấn luyện Qwen3Guard-0.6B..."

# Dùng accelerate để tối ưu train trên GPU
accelerate launch scripts/training/train.py --config configs/qwen3guard_06b_lora_v1.yaml

echo "✅ Hoàn tất quá trình huấn luyện!"
