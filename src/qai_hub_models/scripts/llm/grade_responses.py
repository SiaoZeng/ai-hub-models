#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
r"""CLI for grading LLM responses.

Reads a JSON file containing a list of items in the form::

    [
      {"idx": 0, "prompt": "What is gravity?", "output": "Gravity is ..."},
      ...
    ]

The ``prompt`` is passed through to the grader as-is, and ``output`` is the
generated response to grade. Grading is delegated to
:mod:`qai_hub_models.models._shared.llm.grader`.

Example usage::

    python -m qai_hub_models.scripts.llm.grade_responses responses.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from qai_hub_models.models._shared.llm.grader import (
    DEFAULT_PROMPT_TEMPLATE,
    DESCRIPTIONS,
    LETTER_POINTS,
    LETTERS,
    MAX_POINTS,
    ResponseGrader,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade LLM responses from a JSON file (argmax over A/B/C/D).",
    )
    parser.add_argument(
        "responses_json",
        type=str,
        help="Path to a JSON file: list of {idx, prompt, output} objects.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to grading prompt template (must contain {response}). "
        "If omitted, the default template is used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.6-35B-A3B",
        help="HuggingFace model id to use as grader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for the grader model (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Grader model dtype.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, write a machine-readable summary to this path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-item logits.",
    )
    args = parser.parse_args()

    if args.prompt_file:
        prompt_template = Path(args.prompt_file).read_text()
    else:
        prompt_template = DEFAULT_PROMPT_TEMPLATE

    items = json.loads(Path(args.responses_json).read_text())
    if not items:
        raise ValueError(f"No items found in {args.responses_json}")
    print(f"Loaded {len(items)} items from {args.responses_json}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    grader = ResponseGrader(
        model_id=args.model,
        device=args.device,
        dtype=dtype_map[args.dtype],
        prompt_template=prompt_template,
    )
    print(
        f"Letter token ids: {dict(zip(LETTERS, grader.letter_token_ids, strict=True))}"
    )

    summary = grader.grade(items)

    if args.verbose:
        for item, result in zip(items, summary.results, strict=True):
            print(
                f"  idx={item['idx']}: {result.label}  "
                + "  ".join(f"{l}={result.logits[l]:.2f}" for l in LETTERS)
                + ("  [skipped: empty response]" if result.skipped else "")
            )

    counts = Counter(r.label for r in summary.results)
    items_by_label: dict[str, list[int]] = defaultdict(list)
    for item, result in zip(items, summary.results, strict=True):
        items_by_label[result.label].append(item["idx"])

    print()
    print("=" * 60)
    print(f"Grader: {args.model}")
    print(f"Responses graded: {len(items)}")
    print("=" * 60)
    for letter in LETTERS:
        desc = DESCRIPTIONS[letter]
        pts = LETTER_POINTS[letter]
        print(f"  {letter} [{pts:2d} pts] ({desc}): {counts.get(letter, 0)}")
    print()
    print(
        f"Overall score: {summary.score_pct:.1f}%  "
        f"({summary.total_points}/{MAX_POINTS * len(summary.results)} pts)"
    )
    print()
    for letter in ("A", "B", "C"):
        if items_by_label[letter]:
            print(f"Items scoring {letter} ({DESCRIPTIONS[letter]}):")
            for idx in items_by_label[letter]:
                print(f"  - idx={idx}")
            print()

    if args.output_json:
        out = {
            "input_file": str(args.responses_json),
            "grader_model": args.model,
            "num_items": len(items),
            "score_pct": summary.score_pct,
            "total_points": summary.total_points,
            "max_points": MAX_POINTS * len(summary.results),
            "counts": {letter: counts.get(letter, 0) for letter in LETTERS},
            "items": [
                {
                    "idx": item["idx"],
                    "label": result.label,
                    "points": result.points,
                    "skipped": result.skipped,
                }
                for item, result in zip(items, summary.results, strict=True)
            ],
        }
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"Wrote grading summary to {args.output_json}")


if __name__ == "__main__":
    main()
