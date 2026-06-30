"""
run_generate_pilot.py
─────────────────────
Sinh batch 300–500 mẫu từ generation_jobs.jsonl.

Tính năng:
  • Progress bar realtime (tqdm)
  • Resume-safe: đọc lại job_id đã xong trong output, skip không gọi API lại
  • Parallel với ThreadPoolExecutor (MAX_WORKERS tuỳ chỉnh)
  • Stratified sampling: đảm bảo N job được lấy ĐỀU từ tất cả combo/group (không lấy N đầu tiên)
  • Ghi output ngay sau mỗi request thành công (không mất dữ liệu khi mất điện/internet)
  • Ghi lỗi riêng vào file errors để retry sau

Cách dùng:
  python scripts/run_generate_pilot.py                    # sinh 300 mẫu đầu
  python scripts/run_generate_pilot.py --n 500            # sinh 500 mẫu đầu
  python scripts/run_generate_pilot.py --n 500 --workers 15  # tăng số luồng

Chạy lại sau khi mất kết nối:
  python scripts/run_generate_pilot.py --n 500
  → script tự phát hiện job_id nào đã xong, skip và chạy tiếp phần còn lại.
"""

import json
import os
import sys
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# ── Đường dẫn ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
INPUT      = BASE_DIR / "data" / "generation_jobs.jsonl"
OUTPUT     = BASE_DIR / "data" / "generated_raw.jsonl"       # target của audit script
ERRORS     = BASE_DIR / "data" / "generated_raw_errors.jsonl"

MODEL        = "gpt-4.1-mini"
MAX_RETRIES  = 3        # số lần retry khi lỗi timeout/network
RETRY_DELAY  = 2.0      # giây chờ giữa retry

# ── API Client ────────────────────────────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Lock để ghi file an toàn từ nhiều thread ─────────────────────────────────
_write_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jobs_stratified(path: Path, n: int, seed: int = 42) -> list[dict]:
    """
    Đọc toàn bộ jobs rồi stratified-sample N job trải đều theo 'group'.
    Đảm bảo pilot batch có đủ mọi nhóm: jailbreak, pi, benign, controversial...
    Nếu N >= tổng số job thì trả về toàn bộ (shuffle).
    """
    import random
    rng = random.Random(seed)

    all_jobs: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_jobs.append(json.loads(line))

    if n >= len(all_jobs):
        rng.shuffle(all_jobs)
        return all_jobs

    # Nhóm theo 'group'
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for job in all_jobs:
        groups[job.get("group", "unknown")].append(job)

    # Shuffle từng nhóm
    for g in groups.values():
        rng.shuffle(g)

    # Round-robin lấy từ từng nhóm cho đến khi đủ N
    group_keys    = sorted(groups.keys())
    group_iters   = {k: iter(v) for k, v in groups.items()}
    selected: list[dict] = []

    while len(selected) < n:
        added = False
        for k in group_keys:
            if len(selected) >= n:
                break
            try:
                selected.append(next(group_iters[k]))
                added = True
            except StopIteration:
                pass
        if not added:
            break   # tất cả nhóm đã hết

    rng.shuffle(selected)  # shuffle lại để thứ tự không bị clustered theo group
    return selected


def load_completed_ids(path: Path) -> set[str]:
    """Đọc tập job_id đã hoàn thành trong output file (để resume)."""
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "job_id" in obj:
                    completed.add(obj["job_id"])
            except json.JSONDecodeError:
                pass
    return completed


def extract_text(response) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks).strip()


# ── Generation core ───────────────────────────────────────────────────────────

def call_api(job: dict) -> dict:
    """Gọi API sinh 1 sample. Raise exception nếu thất bại sau MAX_RETRIES."""
    prompt = job["prompt"]
    n_msg  = job["n_messages"]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=MODEL,
                input=prompt,
                timeout=60.0,
                temperature=0.85,
                max_output_tokens=1500,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "guardrail_conversation",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "messages": {
                                    "type": "array",
                                    "minItems": n_msg,
                                    "maxItems": n_msg,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "role":    {"type": "string", "enum": ["user", "assistant"]},
                                            "content": {"type": "string", "minLength": 1}
                                        },
                                        "required": ["role", "content"]
                                    }
                                }
                            },
                            "required": ["messages"]
                        }
                    }
                },
            )
            text = extract_text(response)
            data = json.loads(text)
            return data["messages"]

        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise


def build_sample(job: dict, messages: list) -> dict:
    """Ghép metadata từ job + messages sinh ra thành sample chuẩn."""
    return {
        "job_id":               job["job_id"],
        "sample_id":            job["sample_id"],
        "combo_id":             job["combo_id"],
        "template_id":          job.get("template_id"),
        "language":             job["language"],
        "domain":               job.get("domain"),
        "style":                job.get("style"),
        "n_messages":           job["n_messages"],
        "attack_type":          job.get("attack_type"),
        "scenario":             job.get("scenario"),
        "mechanism":            job.get("mechanism"),
        "evasion":              job.get("evasion"),
        "label":                job["label"],
        "group":                job.get("group"),
        
        # ── Augmentation metadata (Mới thêm) ──
        "base_combo_id":          job.get("base_combo_id"),
        "augmentation_type":      job.get("augmentation_type"),
        "target_error":           job.get("target_error"),
        "augmentation_family_id": job.get("augmentation_family_id"),
        "validation_rule":        job.get("validation_rule"),
        "pair_group":             job.get("pair_group"),
        "variant_type":           job.get("variant_type"),
        "base_sample_id":         job.get("base_sample_id"),
        "positive_constraints":   job.get("positive_constraints"),
        "negative_constraints":   job.get("negative_constraints"),

        "messages":             messages,
        "generation_model":     MODEL,
        "generation_prompt_id": job.get("generation_prompt_id"),
    }


def process_job(job: dict) -> dict:
    """Worker function chạy trong thread pool."""
    try:
        messages = call_api(job)
        sample   = build_sample(job, messages)
        return {"success": True, "job": job, "sample": sample}
    except Exception as e:
        return {"success": False, "job": job, "error": repr(e)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sinh pilot dataset cho Qwen3Guard")
    parser.add_argument("--n",       type=int, default=300, help="Số job cần sinh (default: 300)")
    parser.add_argument("--workers", type=int, default=10,  help="Số luồng song song (default: 10)")
    parser.add_argument("--input",   type=str, default=str(INPUT), help="Đường dẫn file jobs")
    parser.add_argument("--output",  type=str, default=str(OUTPUT), help="Đường dẫn file output")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    errors_path = Path(args.output).parent / (Path(args.output).stem + "_errors.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load jobs ──────────────────────────────────────────────────────────
    print(f"📂 Input : {input_path}")
    print(f"📄 Output: {output_path}")
    print(f"⚙️  Model : {MODEL}  |  Workers: {args.workers}")

    all_jobs = load_jobs_stratified(input_path, args.n)
    print(f"Loaded {len(all_jobs):,} jobs (target n={args.n})")

    # ── 2. Resume: bỏ qua các job đã hoàn thành ──────────────────────────────
    completed_ids = load_completed_ids(output_path)
    if completed_ids:
        print(f"✅ Resume mode: {len(completed_ids):,} job đã hoàn thành, sẽ skip.")

    pending_jobs = [j for j in all_jobs if j["job_id"] not in completed_ids]

    if not pending_jobs:
        print("🎉 Tất cả job đã hoàn thành! Không cần chạy thêm.")
        return

    print(f"▶️  Cần sinh thêm: {len(pending_jobs):,} job\n")

    # ── 3. Chạy song song, ghi file ngay sau mỗi kết quả ─────────────────────
    success_count = 0
    error_count   = 0

    # Mở file ở chế độ APPEND (a) để resume không ghi đè
    with (
        output_path.open("a", encoding="utf-8") as out_f,
        errors_path.open("a", encoding="utf-8") as err_f,
        tqdm(
            total=len(pending_jobs),
            desc="Generating",
            unit="sample",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ) as pbar,
    ):
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(process_job, job): job for job in pending_jobs}

            for future in as_completed(future_map):
                result = future.result()
                job    = result["job"]

                with _write_lock:
                    if result["success"]:
                        # Ghi ngay, flush ngay — đảm bảo không mất dữ liệu
                        out_f.write(json.dumps(result["sample"], ensure_ascii=False) + "\n")
                        out_f.flush()
                        success_count += 1
                        pbar.set_postfix(ok=success_count, err=error_count, refresh=False)
                    else:
                        err_record = {
                            "job_id":    job.get("job_id"),
                            "sample_id": job.get("sample_id"),
                            "combo_id":  job.get("combo_id"),
                            "error":     result["error"],
                        }
                        err_f.write(json.dumps(err_record, ensure_ascii=False) + "\n")
                        err_f.flush()
                        error_count += 1
                        pbar.set_postfix(ok=success_count, err=error_count, refresh=False)

                pbar.update(1)

    # ── 4. Summary ────────────────────────────────────────────────────────────
    total_done = len(completed_ids) + success_count
    print(f"\n{'='*55}")
    print(f"✅ Thành công  : {success_count:,} (lần này) + {len(completed_ids):,} (trước) = {total_done:,} tổng")
    print(f"❌ Lỗi         : {error_count:,} → {errors_path.name}")
    print(f"📄 Output      : {output_path}")
    print(f"{'='*55}")

    if error_count > 0:
        print(f"\n💡 Tip: chạy lại lệnh để retry {error_count} lỗi bị skip.")
        print(f"   Hoặc inspect: {errors_path}")


if __name__ == "__main__":
    main()
