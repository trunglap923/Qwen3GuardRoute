import os
import sys
import yaml
import torch
import argparse
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset


# -----------------------------------------------------------------------------
# CLI / CONFIG
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Custom QLoRA Training Script for Qwen3Guard")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--dry_run", action="store_true", help="Run with few samples and few steps")
    parser.add_argument("--skip_training", action="store_true", help="Only run data + mask debug, then exit")
    parser.add_argument("--debug_max_tokens", type=int, default=260, help="Max tokens to print in mask preview")
    return parser.parse_args()


def load_config(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {yaml_path}")
    return config


def apply_runtime_safety_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Make config safer on Colab/T4-like GPUs."""
    if config.get("bf16", False):
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if not bf16_ok:
            print("⚠️ bf16=True nhưng GPU/runtime không hỗ trợ tốt bf16. Chuyển sang fp16=True, bf16=False.")
            config["bf16"] = False
            config["fp16"] = True
    return config


def apply_dry_run_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    print("⚠️ CHẾ ĐỘ DRY-RUN ĐƯỢC KÍCH HOẠT")
    config["output_dir"] = "outputs/qwen3guard_06b_lora_dryrun"
    config["num_train_epochs"] = 1
    config["max_steps"] = 20
    config["logging_steps"] = 1
    config["eval_steps"] = 10
    config["save_steps"] = 10

    # Safe defaults for Colab T4 16GB.
    config["per_device_train_batch_size"] = 1
    config["per_device_eval_batch_size"] = 1
    config["gradient_accumulation_steps"] = config.get("gradient_accumulation_steps", 4)
    config["bf16"] = False
    config["fp16"] = True

    print(
        "Override Config: "
        f"max_steps={config['max_steps']}, "
        f"train_bs={config['per_device_train_batch_size']}, "
        f"eval_bs={config['per_device_eval_batch_size']}, "
        f"output_dir={config['output_dir']}"
    )
    return config


def save_resolved_config(config: Dict[str, Any]) -> None:
    os.makedirs(config["output_dir"], exist_ok=True)
    resolved_config_path = os.path.join(config["output_dir"], "resolved_config.yaml")
    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    print(f"✅ Saved resolved config: {resolved_config_path}")


# -----------------------------------------------------------------------------
# DATA / MASKING
# -----------------------------------------------------------------------------
def render_chatml_user_prompt(user_content: str) -> str:
    """
    Render ChatML thủ công để tránh tokenizer.apply_chat_template()
    tự inject template gốc của Qwen3Guard như Categories/Refusal.
    """
    return (
        "<|im_start|>user\n"
        + user_content.strip()
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def manual_preprocess_function(examples, tokenizer, max_seq_length):
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    valid_targets = {
        "Safety: Safe",
        "Safety: Controversial",
        "Safety: Unsafe",
    }

    for msgs in examples["messages"]:
        if len(msgs) != 2:
            raise ValueError(f"Expected 2 messages, got {len(msgs)}")

        if msgs[0].get("role") != "user":
            raise ValueError(f"First message must be user, got {msgs[0].get('role')}")

        if msgs[1].get("role") != "assistant":
            raise ValueError(f"Second message must be assistant, got {msgs[1].get('role')}")

        user_content = msgs[0].get("content", "")
        target_text = msgs[1].get("content", "").strip()

        if target_text not in valid_targets:
            raise ValueError(f"Invalid assistant target: {repr(target_text)}")

        prompt_text = render_chatml_user_prompt(user_content)
        full_text = prompt_text + target_text + "<|im_end|>"

        prompt_tokens = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
        )["input_ids"]

        labels = full_tokens.copy()
        prompt_len = len(prompt_tokens)
        labels[:prompt_len] = [-100] * prompt_len

        if len(full_tokens) > max_seq_length:
            full_tokens = full_tokens[:max_seq_length]
            labels = labels[:max_seq_length]

        loss_tokens = [
            tok for tok, lab in zip(full_tokens, labels)
            if lab != -100
        ]

        loss_text = tokenizer.decode(
            loss_tokens,
            skip_special_tokens=True
        ).strip()

        if not loss_text.startswith("Safety:"):
            raise ValueError(
                "Loss masking failed: loss_text does not start with Safety:\n"
                f"target_text={repr(target_text)}\n"
                f"loss_text={repr(loss_text)}"
            )

        if "Categories" in loss_text or "Refusal" in loss_text:
            raise ValueError(
                "Loss masking failed: loss_text contains unwanted template text:\n"
                f"loss_text={repr(loss_text)}"
            )

        input_ids_list.append(full_tokens)
        labels_list.append(labels)
        attention_mask_list.append([1] * len(full_tokens))

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list,
    }


def debug_manual_mask_preview(sample_messages, tokenizer, max_seq_length: int = 2048, max_print_tokens: int = 260) -> bool:
    print("\n" + "=" * 80)
    print("🛠️ DEBUG MANUAL LOSS MASK PREVIEW")
    print("=" * 80)

    batch = {"messages": [sample_messages]}
    out = manual_preprocess_function(batch, tokenizer, max_seq_length)

    input_ids = out["input_ids"][0]
    labels = out["labels"][0]

    loss_token_ids = [tok for tok, lbl in zip(input_ids, labels) if lbl != -100]
    loss_token_count = len(loss_token_ids)
    loss_text = tokenizer.decode(loss_token_ids, skip_special_tokens=True)

    print(f"Total tokens: {len(input_ids)}")
    print(f"Loss token count: {loss_token_count}")
    print(f"Loss text: {repr(loss_text)}")

    if loss_token_count == 0:
        print("❌ LỖI: Không có token nào được tính loss.")
        return False

    if "Safety:" not in loss_text:
        print("❌ LỖI: Phần được tính loss không chứa 'Safety:'.")
        return False

    valid_targets = ["Safety: Safe", "Safety: Controversial", "Safety: Unsafe"]
    if not any(t in loss_text for t in valid_targets):
        print("⚠️ CẢNH BÁO: Loss text có 'Safety:' nhưng không khớp hoàn toàn 1 trong 3 target chuẩn.")
        print(f"Expected one of: {valid_targets}")

    print("\nPreview tokens:")
    print(f"{'TYPE':<10} | {'TOKEN_ID':<10} | {'TOKEN_TEXT'}")
    print("-" * 60)

    n = min(len(input_ids), max_print_tokens)
    for tok_id, lbl_id in zip(input_ids[:n], labels[:n]):
        token_str = tokenizer.decode([tok_id])
        if lbl_id == -100:
            print(f"[MASK]     | {tok_id:<10} | {repr(token_str)}")
        else:
            print(f"[LOSS]     | {tok_id:<10} | {repr(token_str)}")

    if len(input_ids) > n:
        print(f"... truncated preview: printed {n}/{len(input_ids)} tokens ...")
        print("Loss tokens at the end:")
        for tok_id in loss_token_ids:
            print(f"[LOSS]     | {tok_id:<10} | {repr(tokenizer.decode([tok_id]))}")

    print("=" * 80 + "\n")
    return True


@dataclass
class CausalLMCollator:
    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must have pad_token_id or eos_token_id")

        max_len = max(len(f["input_ids"]) for f in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            input_ids = list(f["input_ids"])
            attention_mask = list(f["attention_mask"])
            labels = list(f["labels"])

            pad_len = max_len - len(input_ids)
            input_ids = input_ids + [pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [self.label_pad_token_id] * pad_len

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------
def build_training_args(config: Dict[str, Any]) -> TrainingArguments:
    """Build TrainingArguments with compatibility for eval_strategy/evaluation_strategy."""
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    kwargs = dict(
        output_dir=config["output_dir"],
        learning_rate=config.get("learning_rate", 2e-5),
        num_train_epochs=config.get("num_train_epochs", 3.0),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        eval_steps=config.get("eval_steps", 100),
        save_steps=config.get("save_steps", 100),
        logging_steps=config.get("logging_steps", 10),
        save_total_limit=config.get("save_total_limit", 3),
        bf16=config.get("bf16", False),
        fp16=config.get("fp16", False),
        max_steps=config.get("max_steps", -1),
        load_best_model_at_end=config.get("load_best_model_at_end", False),
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        seed=config.get("seed", 42),
        report_to=config.get("report_to", "none"),
        remove_unused_columns=False,
    )

    eval_value = config.get("eval_strategy", config.get("evaluation_strategy", "steps"))
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_value
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = eval_value

    if "gradient_checkpointing" in params:
        kwargs["gradient_checkpointing"] = config.get("gradient_checkpointing", False)

    # Drop kwargs not supported by the installed transformers version.
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    return TrainingArguments(**kwargs)


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.dry_run:
        config = apply_dry_run_overrides(config)

    config = apply_runtime_safety_overrides(config)
    save_resolved_config(config)

    # 1. Tokenizer
    print(f"⏳ Đang tải Tokenizer: {config['model_name_or_path']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Dataset
    print(f"⏳ Đang tải Dataset: {config['train_file']}")
    train_dataset = load_dataset("json", data_files=config["train_file"], split="train")
    eval_dataset = load_dataset("json", data_files=config["eval_file"], split="train")

    if args.dry_run:
        train_dataset = train_dataset.select(range(min(50, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(20, len(eval_dataset))))

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Save raw eval sample BEFORE map(), because map(remove_columns=...) removes messages.
    sanity_messages = eval_dataset[0]["messages"]
    sanity_target = sanity_messages[-1]["content"]

    # 3. Debug masking before loading model.
    max_seq = int(config.get("max_seq_length", 2048))
    sample_messages = train_dataset[0]["messages"]
    is_mask_valid = debug_manual_mask_preview(
        sample_messages,
        tokenizer,
        max_seq_length=max_seq,
        max_print_tokens=args.debug_max_tokens,
    )

    if not is_mask_valid:
        print("❌ Debug Mask thất bại. Dừng script.")
        sys.exit(1)

    if args.skip_training:
        print("🛑 --skip_training được bật. Dừng script sau khi debug mask PASS.")
        return

    # 4. Model + QLoRA
    print("⏳ Đang tải QLoRA Model (4-bit)...")
    compute_dtype = torch.bfloat16 if config.get("bf16", False) else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.get("load_in_4bit", True),
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.get("gradient_checkpointing", False),
    )

    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("lora_target_modules", "all-linear"),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Preprocess dataset.
    print("🔧 Đang preprocess dataset bằng manual assistant-only loss masking...")
    train_dataset = train_dataset.map(
        lambda x: manual_preprocess_function(x, tokenizer, max_seq),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )
    eval_dataset = eval_dataset.map(
        lambda x: manual_preprocess_function(x, tokenizer, max_seq),
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval dataset",
    )

    collator = CausalLMCollator(tokenizer=tokenizer, label_pad_token_id=-100)

    # 6. Trainer setup.
    training_args = build_training_args(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # 7. Train & Save.
    print("🚀 BẮT ĐẦU HUẤN LUYỆN...")
    trainer.train()

    print("💾 Đang lưu Adapter và Tokenizer...")
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    # 8. Sanity inference after dry-run.
    if args.dry_run:
        print("\n" + "=" * 50)
        print("🧠 SANITY CHECK INFERENCE SAU DRY-RUN")
        print("=" * 50)

        model.eval()
        user_content = sanity_messages[0]["content"]
        test_prompt = render_chatml_user_prompt(user_content)
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            result = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

        first_line = result.splitlines()[0].strip() if result else ""

        print(f"Target Expected: {sanity_target}")
        print(f"Model Generated Raw: {repr(result)}")
        print(f"Model First Line: {first_line}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()