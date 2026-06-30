#!/bin/bash
echo "🚀 Bắt đầu cài đặt môi trường Qwen3Guard trên Vast AI..."

# 1. Cập nhật apt-get và cài đặt các công cụ cơ bản
apt-get update && apt-get install -y unzip htop tmux git jq

# 2. Cài đặt Python dependencies
echo "📦 Cài đặt thư viện Python..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Khắc phục triệt để lỗi thư viện CUDA (Bitsandbytes 4-bit)
echo "🔧 Tự động biên dịch bitsandbytes để tương thích 100% với CUDA của máy chủ..."
pip uninstall -y bitsandbytes
pip install cmake
CURRENT_DIR=$(pwd)
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git /tmp/bitsandbytes_src
cd /tmp/bitsandbytes_src
cmake -DCOMPUTE_BACKEND=cuda -S . -B build
cmake --build build -j$(nproc)
pip install .
cd "$CURRENT_DIR"
rm -rf /tmp/bitsandbytes_src
echo "✅ Đã nạp thành công bộ thư viện C++ lõi của Native CUDA!"

# 3. Log in HuggingFace (nếu cần tải model private)
# uncomment and add token if needed
# huggingface-cli login --token "YOUR_HF_TOKEN" --add-to-git-credential

# 4. Log in WandB (để tracking quá trình train)
# uncomment and add token if needed
# wandb login "YOUR_WANDB_TOKEN"

chmod +x train.sh eval.sh

echo "✅ Cài đặt hoàn tất! Bạn có thể bắt đầu quá trình train bằng lệnh:"
echo "./train.sh"
