
#!/usr/bin/env python3
"""Evaluate a base model or LoRA student adapter on a held-out JSONL test dataset."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from statistics import mean

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger("evaluate_student")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a base model or trained LoRA student adapter."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/distillation/configs/cpu.yaml"),
    )

    parser.add_argument(
        "--adapter-dir",
        type=Path,
        help="Directory containing the trained LoRA adapter.",
    )

    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Evaluate the untouched base model without loading a LoRA adapter.",
    )

    parser.add_argument(
        "--test-file",
        type=Path,
        help="Override the test dataset path from the config.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Override the evaluation results JSONL path.",
    )

    parser.add_argument("--max-new-tokens", type=int, default=256)

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing results file.",
    )

    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    return config


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )

            for field in ("id", "prompt", "response"):
                if field not in record:
                    raise ValueError(
                        f"Missing '{field}' on line {line_number}."
                    )

            if (
                not isinstance(record["prompt"], str)
                or not record["prompt"].strip()
            ):
                raise ValueError(f"Invalid prompt on line {line_number}.")

            if (
                not isinstance(record["response"], str)
                or not record["response"].strip()
            ):
                raise ValueError(
                    f"Invalid reference response on line {line_number}."
                )

            records.append(record)

    if not records:
        raise ValueError("The test dataset contains no usable records.")

    return records


def format_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"User: {prompt}\nAssistant:"


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[str, int, float]:

    formatted_prompt = format_prompt(tokenizer, prompt)

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[1]

    start_time = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    generated_ids = output[0, input_length:]
    generated_tokens = int(generated_ids.numel())

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return answer, generated_tokens, elapsed


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive.")

    if args.base_only and args.adapter_dir:
        raise ValueError(
            "--base-only cannot be combined with --adapter-dir."
        )

    config = load_config(args.config)

    base_model_name = config["model_name"]

    if args.base_only:
        adapter_dir = None
        default_output_dir = Path(
            "examples/distillation/outputs/base-only"
        )
        evaluation_mode = "base_model_only"
    else:
        adapter_dir = (
            args.adapter_dir
            or Path(config["output"]["directory"])
        )
        default_output_dir = adapter_dir
        evaluation_mode = "lora_adapter"

    test_file = (
        args.test_file
        or Path(config["dataset"]["test_file"])
    )

    output_file = (
        args.output
        or default_output_dir / "evaluation-results.jsonl"
    )

    if not args.base_only:
        if not adapter_dir.is_dir():
            raise FileNotFoundError(
                f"Adapter directory not found: {adapter_dir}"
            )

        if not (adapter_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(
                "adapter_config.json is missing."
            )

        if not (adapter_dir / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(
                "adapter_model.safetensors is missing."
            )

    summary_file = output_file.with_name(
        "evaluation-summary.json"
    )

    if output_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"Results already exist: {output_file}. "
            "Use --overwrite to replace them."
        )

    if summary_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"Summary already exists: {summary_file}. "
            "Use --overwrite to replace it."
        )

    records = read_jsonl(test_file)

    device = select_device(args.device)

    model_dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    LOGGER.info("Base model: %s", base_model_name)
    LOGGER.info("Evaluation mode: %s", evaluation_mode)
    LOGGER.info("Adapter: %s", adapter_dir)
    LOGGER.info("Test records: %s", len(records))
    LOGGER.info("Device: %s", device)
    LOGGER.info("Loading tokenizer...")

    if (
        not args.base_only
        and (adapter_dir / "tokenizer_config.json").is_file()
    ):
        tokenizer_source = str(adapter_dir)
    else:
        tokenizer_source = base_model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=model_dtype,
    )

    if args.base_only:
        model = base_model
        LOGGER.info(
            "Using untouched base model; no adapter loaded."
        )
    else:
        LOGGER.info("Loading trained LoRA adapter...")

        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_dir),
            is_trainable=False,
        )

    model.to(device)
    model.eval()

    LOGGER.info("Starting held-out test evaluation.")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = []
    total_generated_tokens = 0
    total_generation_time = 0.0

    with output_file.open("w", encoding="utf-8") as output:
        for index, record in enumerate(records, start=1):
            LOGGER.info(
                "Evaluating %s/%s: %s",
                index,
                len(records),
                record["id"],
            )

            answer, token_count, elapsed = generate_answer(
                model=model,
                tokenizer=tokenizer,
                prompt=record["prompt"],
                device=device,
                max_new_tokens=args.max_new_tokens,
            )

            result = {
                "id": record["id"],
                "task": record.get("task"),
                "prompt": record["prompt"],
                "teacher_reference": record["response"],
                "student_response": answer,
                "generated_tokens": token_count,
                "generation_time_seconds": round(elapsed, 4),
                "generation_tokens_per_second": (
                    round(token_count / elapsed, 3)
                    if elapsed > 0
                    else None
                ),
                "teacher_model": record.get("teacher_model"),
                "student_base_model": base_model_name,
                "evaluation_method": "greedy_generation",
                "evaluation_mode": evaluation_mode,
            }

            output.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            output.flush()

            results.append(result)

            total_generated_tokens += token_count
            total_generation_time += elapsed

            LOGGER.info(
                "Generated %s tokens in %.2f seconds.",
                token_count,
                elapsed,
            )

    mean_time = mean(
        item["generation_time_seconds"]
        for item in results
    )

    summary = {
        "evaluation_mode": evaluation_mode,
        "base_model": base_model_name,
        "adapter_directory": (
            str(adapter_dir)
            if adapter_dir is not None
            else None
        ),
        "test_records": len(results),
        "device": str(device),
        "max_new_tokens": args.max_new_tokens,
        "total_generated_tokens": total_generated_tokens,
        "total_generation_time_seconds": round(
            total_generation_time,
            4,
        ),
        "average_generation_time_seconds": round(
            mean_time,
            4,
        ),
        "overall_generation_tokens_per_second": (
            round(
                total_generated_tokens / total_generation_time,
                3,
            )
            if total_generation_time > 0
            else None
        ),
        "results_file": str(output_file),
        "note": (
            "Generation statistics do not establish answer quality. "
            "Review student responses against references independently."
        ),
    }

    with summary_file.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    LOGGER.info("Evaluation complete.")
    LOGGER.info("Results: %s", output_file)
    LOGGER.info("Summary: %s", summary_file)
    LOGGER.info(
        "Average generation time: %.2f seconds",
        mean_time,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Evaluation failed.")
        raise
