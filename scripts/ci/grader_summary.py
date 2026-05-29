#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Render a markdown table summarizing LLM grader output JSON files.

Reads ``*_grade.json`` files (produced by
``qai_hub_models.scripts.llm.grade_responses --output-json``) from the given
directory and prints a markdown table to stdout. Intended for use in a
GitHub Actions step that appends to ``$GITHUB_STEP_SUMMARY``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Directory containing *_grade.json files.")
    args = parser.parse_args()

    rows: list[str] = []
    for path in sorted(glob.glob(os.path.join(args.directory, "*_grade.json"))):
        with open(path) as f:
            d = json.load(f)
        name = os.path.basename(d["input_file"]).replace("_eval.json", "")
        c = d["counts"]
        rows.append(
            f"| {name} | {d['score_pct']:.1f}% | "
            f"{c.get('A', 0)} | {c.get('B', 0)} | {c.get('C', 0)} | {c.get('D', 0)} |"
        )

    if not rows:
        return

    print("## LLM Response Grading")
    print()
    print("| File | Score | A | B | C | D |")
    print("|------|------:|--:|--:|--:|--:|")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
