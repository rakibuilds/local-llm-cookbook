#!/usr/bin/env python3
"""Train a local student LLM using response-based distillation and LoRA."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import torch
    import yaml
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
except ImportError as exc:
    print(
        f"Missing dependency: {exc}\n"
        "Install the appropriate PyTorch build and project requirements first.",
        file=sys.stderr,
    )
    raise SystemExit(1)


LOGGER = logging.getLogger("distillation")
REPO_ROOT = Path(__file__).resolve().parents[2]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    for section in ("dataset", "output", "training", "lora"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing or invalid configuration section: {section}")

    if not isinstance(config.get("model_name"), str) or not config["model_name"].strip():
        raise ValueError("model_name must be a non-empty string.")

    return config


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}, line {line_number}."
                )

            record_id = record.get("id")
            if isinstance(record_id, bool) or not isinstance(record_id, (str, int)):
                raise ValueError(
                    f"Invalid record ID in {path}, line {line_number}."
                )

            for field in ("prompt", "response"):
                value = record.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Missing or empty '{field}' in {path}, line {line_number}."
                    )

            records.append(record)

    if not records:
        raise ValueError(f"No usable records found in {path}")

    return records


def check_split_integrity(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    test_path: Path,
) -> None:
    train_ids = {str(record["id"]) for record in train_records}
    validation_ids = {str(record["id"]) for record in validation_records}

    if train_ids.intersection(validation_ids):
        raise ValueError("Training and validation datasets contain overlapping IDs.")

    if not test_path.is_file():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")

    test_records = read_jsonl(test_path)
    test_ids = {str(record["id"]) for record in test_records}

    if train_ids.intersection(test_ids):
        raise ValueError("Training and test datasets contain overlapping IDs.")

    if validation_ids.intersection(test_ids):
        raise ValueError("Validation and test datasets contain overlapping IDs.")

    LOGGER.info(
        "Dataset split integrity passed: %d train, %d validation, %d test records.",
        len(train_records),
        len(validation_records),
        len(test_records),
    )
    LOGGER.warning(
        "The test split is checked for overlap only. It will not be used for training."
    )


class DistillationDataset(Dataset):
    """Tokenized prompt/response examples with response-only training labels."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.examples: list[dict[str, list[int]]] = []
        self.max_length = max_length

        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad token.")

        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token.")

        for record in records:
            prompt = record["prompt"].strip()
            response = record["response"].strip()

            prompt_messages = [{"role": "user", "content": prompt}]
            full_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]

            if getattr(tokenizer, "chat_template", None):
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                full_text = tokenizer.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                if not full_text.startswith(prompt_text):
                    raise ValueError(
                        "Chat template does not preserve the prompt prefix. "
                        "Cannot safely mask prompt tokens."
                    )
            else:
                # Fallback for models without a chat template.
                prompt_text = f"User:\n{prompt}\n\nAssistant:\n"
                full_text = f"{prompt_text}{response}{tokenizer.eos_token}"

            prefix_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            full_ids = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            # Find the token prefix shared by the prompt and complete example.
            common_length = 0
            for prefix_token, full_token in zip(prefix_ids, full_ids):
                if prefix_token != full_token:
                    break
                common_length += 1

            if common_length == 0 or common_length >= len(full_ids):
                raise ValueError(
                    f"Could not identify response tokens for record {record['id']}."
                )

            response_ids = full_ids[common_length:]
            prefix_ids = full_ids[:common_length]

            # Preserve some prompt context while reserving space for the response.
            prompt_budget = max(1, max_length // 4)
            if len(prefix_ids) > max_length - prompt_budget:
                prefix_ids = prefix_ids[-(max_length - prompt_budget):]

            response_budget = max_length - len(prefix_ids)
            response_ids = response_ids[:response_budget]

            if not response_ids:
                raise ValueError(
                    f"Record {record['id']} has no response tokens after truncation."
                )

            input_ids = prefix_ids + response_ids
            labels = ([-100] * len(prefix_ids)) + response_ids.copy()

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )

        if not self.examples:
            raise ValueError("Tokenization produced no training examples.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class ResponseOnlyCollator:
    """Right-pad input tokens and mask padded labels from the loss."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        features: list[dict[str, list[int]]],
    ) -> dict[str, torch.Tensor]:
        max_length = max(len(item["input_ids"]) for item in features)

        input_ids = []
        attention_masks = []
        labels = []

        for item in features:
            padding_length = max_length - len(item["input_ids"])

            input_ids.append(
                item["input_ids"] + [self.pad_token_id] * padding_length
            )
            attention_masks.append(
                item["attention_mask"] + [0] * padding_length
            )
            labels.append(item["labels"] + [-100] * padding_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def validate_training_config(config: dict[str, Any]) -> None:
    training = config["training"]
    lora = config["lora"]

    positive_values = (
        "num_train_epochs",
        "max_seq_length",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "save_total_limit",
    )

    for key in positive_values:
        value = training.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"training.{key} must be greater than zero.")

    if training.get("fp16", False) and training.get("bf16", False):
        raise ValueError("Enable either fp16 or bf16, not both.")

    if lora.get("enabled", True):
        for key in ("rank", "alpha"):
            value = lora.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"lora.{key} must be a positive integer.")

        if not lora.get("target_modules"):
            raise ValueError("LoRA target_modules cannot be empty.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a student language model using response-based distillation."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Configuration YAML path, relative to the repository root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and datasets without downloading or training a model.",
    )
    args = parser.parse_args()

    configure_logging()

    config = load_config(Path(args.config))
    validate_training_config(config)

    training = config["training"]
    dataset_config = config["dataset"]
    lora_config = config["lora"]

    train_path = resolve_path(dataset_config["train_file"])
    validation_path = resolve_path(dataset_config["validation_file"])
    test_path = resolve_path(dataset_config["test_file"])
    output_dir = resolve_path(config["output"]["directory"])

    train_records = read_jsonl(train_path)
    validation_records = read_jsonl(validation_path)

    check_split_integrity(train_records, validation_records, test_path)

    LOGGER.info("Model: %s", config["model_name"])
    LOGGER.info("Training examples: %d", len(train_records))
    LOGGER.info("Validation examples: %d", len(validation_records))
    LOGGER.info("Output directory: %s", output_dir)

    if args.dry_run:
        LOGGER.info("Dry run passed. No model was downloaded and no training was started.")
        return 0

    if training.get("fp16", False) or training.get("bf16", False):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "This configuration requests GPU mixed precision, "
                "but CUDA is unavailable. Use configs/cpu.yaml on this machine."
            )

    if training.get("bf16", False) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16.")

    seed = int(training.get("seed", 42))
    set_seed(seed)

    if torch.cuda.is_available():
        LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(0))
        LOGGER.info(
            "GPU memory: %.2f GiB",
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
        )
    else:
        LOGGER.info("CUDA unavailable. Training will use the CPU.")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    max_length = int(training["max_seq_length"])

    train_dataset = DistillationDataset(
        train_records, tokenizer, max_length
    )
    validation_dataset = DistillationDataset(
        validation_records, tokenizer, max_length
    )

    if training.get("fp16", False):
        model_dtype = torch.float16
    elif training.get("bf16", False):
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        dtype=model_dtype,
    )

    if training.get("gradient_checkpointing", False):
        model.config.use_cache = False

    if lora_config.get("enabled", True):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_config["rank"]),
            lora_alpha=int(lora_config["alpha"]),
            lora_dropout=float(lora_config.get("dropout", 0.0)),
            target_modules=lora_config["target_modules"],
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(training["per_device_train_batch_size"])
    gradient_accumulation = int(training["gradient_accumulation_steps"])

    batches_per_epoch = math.ceil(len(train_dataset) / batch_size)
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / gradient_accumulation
    )
    total_optimizer_steps = math.ceil(
        optimizer_steps_per_epoch * float(training["num_train_epochs"])
    )
    warmup_steps = int(
        total_optimizer_steps * float(training.get("warmup_ratio", 0.0))
    )

    LOGGER.info(
        "Training steps: %s | Warmup steps: %s",
        total_optimizer_steps,
        warmup_steps,
    )
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training["num_train_epochs"]),
        "per_device_train_batch_size": int(
            training["per_device_train_batch_size"]
        ),
        "per_device_eval_batch_size": int(
            training["per_device_eval_batch_size"]
        ),
        "gradient_accumulation_steps": int(
            training["gradient_accumulation_steps"]
        ),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "warmup_steps": warmup_steps,
        "logging_steps": int(training["logging_steps"]),
        "save_strategy": training["save_strategy"],
        "save_total_limit": int(training["save_total_limit"]),
        "fp16": bool(training.get("fp16", False)),
        "bf16": bool(training.get("bf16", False)),
        "gradient_checkpointing": bool(
            training.get("gradient_checkpointing", False)
        ),
        "dataloader_num_workers": int(
            training.get("dataloader_num_workers", 0)
        ),
        "seed": seed,
        "report_to": "none",
        "remove_unused_columns": False,
    }

    # Transformers has used both names across releases.
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        training_kwargs["eval_strategy"] = training["eval_strategy"]
    elif "evaluation_strategy" in parameters:
        training_kwargs["evaluation_strategy"] = training["eval_strategy"]
    else:
        raise RuntimeError(
            "Installed Transformers version does not expose an evaluation strategy."
        )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=ResponseOnlyCollator(tokenizer.pad_token_id),
    )

    LOGGER.info("Starting response-based distillation training.")
    result = trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "model_name": config["model_name"],
        "training_method": "response-based distillation with LoRA",
        "training_records": len(train_records),
        "validation_records": len(validation_records),
        "test_records_used_for_training": False,
        "seed": seed,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "train_metrics": result.metrics,
        "torch_version": torch.__version__,
    }

    with (output_dir / "training_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    LOGGER.info("Training complete. Saved artifacts to: %s", output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        logging.error("%s", exc)
        raise SystemExit(1)


