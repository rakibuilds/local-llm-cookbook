#!/usr/bin/env python3
"""
Generate teacher responses using a local Ollama model.

Input:  JSONL records containing id, task, and prompt.
Output: JSONL records containing id, task, prompt, response, and teacher_model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 300
DEFAULT_RETRIES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate teacher responses using a local Ollama model."
    )

    parser.add_argument("--model", required=True, help="Ollama model name.")
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL file.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL file.")
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Ollama host (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Retries for failed requests (default: {DEFAULT_RETRIES}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7).",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant. Give accurate, clear, and relevant answers.",
        help="System instruction for the teacher model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )

    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero.")

    if args.retries < 0:
        parser.error("--retries cannot be negative.")

    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2.")

    return args


def load_prompts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: list[dict[str, Any]] = []

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

            prompt = record.get("prompt")

            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"Line {line_number} must contain a non-empty string 'prompt'."
                )

            task = record.get("task", "general")
            record_id = record.get("id", f"prompt-{line_number:04d}")

            if not isinstance(task, str) or not task.strip():
                raise ValueError(
                    f"Line {line_number} has an invalid 'task' field."
                )

            if not isinstance(record_id, (str, int)):
                raise ValueError(
                    f"Line {line_number} has an invalid 'id' field."
                )

            records.append(
                {
                    "id": str(record_id),
                    "task": task,
                    "prompt": prompt.strip(),
                }
            )

    if not records:
        raise ValueError("The input file contains no valid prompt records.")

    return records


def generate_response(
    session: requests.Session,
    args: argparse.Namespace,
    prompt: str,
) -> str:
    url = f"{args.host.rstrip('/')}/api/chat"

    payload = {
        "model": args.model,
        "stream": False,
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": args.temperature,
        },
    }

    for attempt in range(args.retries + 1):
        try:
            response = session.post(
                url,
                json=payload,
                timeout=args.timeout,
            )
            response.raise_for_status()

            data = response.json()
            message = data.get("message", {})
            content = message.get("content")

            if not isinstance(content, str) or not content.strip():
                raise ValueError("Ollama returned an empty or invalid response.")

            return content.strip()

        except (requests.RequestException, ValueError) as exc:
            if attempt >= args.retries:
                raise RuntimeError(
                    f"Teacher generation failed after {attempt + 1} attempt(s): {exc}"
                ) from exc

            wait_seconds = min(2 ** attempt, 10)
            print(
                f"Request failed: {exc}. Retrying in {wait_seconds}s...",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Unexpected generation failure.")


def main() -> int:
    args = parse_args()

    try:
        records = load_prompts(args.input)

        if args.input.resolve() == args.output.resolve():
            raise ValueError("Input and output files must be different.")

        if args.output.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {args.output}. "
                "Use --overwrite to replace it."
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)

        print(f"Teacher model: {args.model}")
        print(f"Prompts loaded: {len(records)}")
        print(f"Ollama host: {args.host}")

        with requests.Session() as session:
            with args.output.open("w", encoding="utf-8") as output_file:
                for index, record in enumerate(records, start=1):
                    print(
                        f"[{index}/{len(records)}] Generating: {record['id']}"
                    )

                    answer = generate_response(
                        session=session,
                        args=args,
                        prompt=record["prompt"],
                    )

                    result = {
                        **record,
                        "response": answer,
                        "teacher_model": args.model,
                    }

                    output_file.write(
                        json.dumps(result, ensure_ascii=False) + "\n"
                    )
                    output_file.flush()

        print(f"\nCompleted successfully: {args.output}")
        print(f"Generated responses: {len(records)}")
        return 0

    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())