#!/usr/bin/env python3
"""
Validate and prepare response-based LLM distillation datasets.

Expected input fields:
    id, task, prompt, response, teacher_model

Checks:
    - JSONL syntax and record structure
    - Required fields and field types
    - Empty values
    - Duplicate task/prompt pairs
    - Unusually short responses
    - Dataset statistics
    - Optional clean output and train/validation/test splits

This tool checks structure and basic quality signals.
It does not verify factual correctness or training suitability.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "id",
    "task",
    "prompt",
    "response",
    "teacher_model",
)

STRING_FIELDS = (
    "task",
    "prompt",
    "response",
    "teacher_model",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and optionally split a distillation JSONL dataset."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input JSONL dataset.",
    )
    parser.add_argument(
        "--clean-output",
        type=Path,
        help="Optional path for valid, deduplicated records.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        help="Training split output.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        help="Validation split output.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        help="Test split output.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training split ratio (default: 0.8).",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--min-response-chars",
        type=int,
        default=20,
        help="Warn about responses shorter than this length.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output files.",
    )

    args = parser.parse_args()

    ratios = (
        args.train_ratio,
        args.validation_ratio,
        args.test_ratio,
    )

    if any(ratio < 0 or ratio > 1 for ratio in ratios):
        parser.error("Split ratios must be between 0 and 1.")

    if abs(sum(ratios) - 1.0) > 1e-9:
        parser.error(
            "Train, validation, and test ratios must sum to 1."
        )

    if args.min_response_chars < 0:
        parser.error("--min-response-chars cannot be negative.")

    split_paths = (
        args.train_output,
        args.validation_output,
        args.test_output,
    )

    if any(split_paths) and not all(split_paths):
        parser.error(
            "Provide all three split paths: "
            "--train-output, --validation-output, and --test-output."
        )

    return args


def load_dataset(
    path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    records: list[dict[str, Any]] = []
    errors: list[str] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                errors.append(
                    f"Line {line_number}: invalid JSON ({exc.msg})."
                )
                continue

            if not isinstance(record, dict):
                errors.append(
                    f"Line {line_number}: expected a JSON object."
                )
                continue

            missing = [
                field
                for field in REQUIRED_FIELDS
                if field not in record
            ]

            if missing:
                errors.append(
                    f"Line {line_number}: missing fields: "
                    + ", ".join(missing)
                )
                continue

            record_id = record["id"]

            if (
                isinstance(record_id, bool)
                or not isinstance(record_id, (str, int))
                or not str(record_id).strip()
            ):
                errors.append(
                    f"Line {line_number}: 'id' must be a "
                    "non-empty string or integer."
                )
                continue

            invalid_fields = [
                field
                for field in STRING_FIELDS
                if not isinstance(record[field], str)
                or not record[field].strip()
            ]

            if invalid_fields:
                errors.append(
                    f"Line {line_number}: empty or invalid fields: "
                    + ", ".join(invalid_fields)
                )
                continue

            records.append(record)

    return records, errors


def validate_records(
    records: list[dict[str, Any]],
    min_response_chars: int,
) -> tuple[list[dict[str, Any]], list[str], int]:

    valid_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()
    duplicate_count = 0

    for record in records:

        prompt = record["prompt"].strip()
        response = record["response"].strip()
        task = record["task"].strip()

        duplicate_key = (
            task.casefold(),
            prompt.casefold(),
        )

        if duplicate_key in seen:
            duplicate_count += 1

            warnings.append(
                f"Record {record['id']}: duplicate task/prompt pair."
            )
            continue

        seen.add(duplicate_key)

        if len(response) < min_response_chars:
            warnings.append(
                f"Record {record['id']}: response is unusually short "
                f"({len(response)} characters)."
            )

        cleaned_record = dict(record)

        cleaned_record["id"] = str(record["id"]).strip()
        cleaned_record["task"] = task
        cleaned_record["prompt"] = prompt
        cleaned_record["response"] = response
        cleaned_record["teacher_model"] = (
            record["teacher_model"].strip()
        )

        valid_records.append(cleaned_record)

    return valid_records, warnings, duplicate_count


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def split_dataset(
    records: list[dict[str, Any]],
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)

    train_end = int(total * train_ratio)
    validation_end = (
        train_end + int(total * validation_ratio)
    )

    train_records = shuffled[:train_end]
    validation_records = shuffled[
        train_end:validation_end
    ]
    test_records = shuffled[validation_end:]

    return (
        train_records,
        validation_records,
        test_records,
    )


def check_output_paths(
    args: argparse.Namespace,
) -> None:

    outputs = [
        args.clean_output,
        args.train_output,
        args.validation_output,
        args.test_output,
    ]

    paths = [
        path.resolve()
        for path in outputs
        if path is not None
    ]

    if len(paths) != len(set(paths)):
        raise ValueError(
            "Output paths must be different."
        )

    input_path = args.input.resolve()

    if input_path in paths:
        raise ValueError(
            "An output path cannot overwrite the input dataset."
        )

    existing = [
        path
        for path in outputs
        if path is not None and path.exists()
    ]

    if existing and not args.overwrite:
        raise FileExistsError(
            "Output file(s) already exist: "
            + ", ".join(str(path) for path in existing)
            + ". Use --overwrite to replace them."
        )


def main() -> int:
    args = parse_args()

    try:
        check_output_paths(args)

        records, errors = load_dataset(args.input)

        valid_records, warnings, duplicate_count = (
            validate_records(
                records,
                args.min_response_chars,
            )
        )

        print("\nDISTILLATION DATASET VALIDATION")
        print("=" * 45)

        print(f"Input file:          {args.input}")
        print(f"Valid JSON records:  {len(records)}")
        print(f"Schema errors:       {len(errors)}")
        print(f"Duplicate records:   {duplicate_count}")
        print(f"Usable records:      {len(valid_records)}")

        if errors:
            print("\nERRORS")

            for error in errors:
                print(f"  - {error}")

        if warnings:
            print("\nWARNINGS")

            for warning in warnings:
                print(f"  - {warning}")

        if valid_records:
            task_counts = Counter(
                record["task"]
                for record in valid_records
            )

            teacher_counts = Counter(
                record["teacher_model"]
                for record in valid_records
            )

            response_lengths = [
                len(record["response"])
                for record in valid_records
            ]

            print("\nDATASET STATISTICS")

            print(f"Tasks:               {dict(task_counts)}")
            print(f"Teacher models:      {dict(teacher_counts)}")

            print(
                "Average response:    "
                f"{sum(response_lengths) / len(response_lengths):.1f} "
                "characters"
            )

            print(
                f"Shortest response:   {min(response_lengths)} characters"
            )

            print(
                f"Longest response:    {max(response_lengths)} characters"
            )

        if args.clean_output is not None:
            write_jsonl(
                args.clean_output,
                valid_records,
            )

            print(
                f"\nClean dataset saved: {args.clean_output}"
            )

        split_requested = args.train_output is not None

        if split_requested:

            if errors or not valid_records:
                raise ValueError(
                    "Cannot create training splits while schema "
                    "errors exist or no usable records remain. "
                    "Fix the dataset first."
                )

            if len(valid_records) < 3:
                raise ValueError(
                    "At least three usable records are required "
                    "to create non-empty train, validation, "
                    "and test splits."
                )

            train, validation, test = split_dataset(
                valid_records,
                args.train_ratio,
                args.validation_ratio,
                args.seed,
            )

            if not train or not validation or not test:
                raise ValueError(
                    "The dataset is too small for the requested "
                    "split ratios. Add more records or adjust "
                    "the ratios."
                )

            write_jsonl(args.train_output, train)
            write_jsonl(args.validation_output, validation)
            write_jsonl(args.test_output, test)

            print("\nDATASET SPLITS")
            print(f"Training:           {len(train)}")
            print(f"Validation:         {len(validation)}")
            print(f"Test:               {len(test)}")

        print("\nValidation finished.")

        if errors or not valid_records:

            if not valid_records and not errors:
                print(
                    "ERROR: Dataset contains no usable records."
                )

            print(
                "Result: FAILED — fix dataset issues before training."
            )

            return 1

        if warnings:
            print(
                "Result: PASSED WITH WARNINGS — "
                "review before training."
            )
        else:
            print("Result: PASSED")

        return 0

    except (OSError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())