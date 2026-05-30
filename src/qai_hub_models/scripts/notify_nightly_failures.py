#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Generate GitHub issue bodies for nightly workflow failures.

Splits failures into two categories and writes issue bodies to an
output directory:
  - workbench_issue.md + workbench_title.txt (AI Hub compile/profile/inference/link)
  - general_issue.md + general_title.txt (unit tests, pre-commit, model tests, etc.)

The workflow shell step then uses `gh issue create --body-file` to
push them to GitHub.

Failed jobs are passed as CLI arguments (gathered by the workflow shell
step via `gh api`), following the same pattern as generate_test_summary.

Usage:
    python -m qai_hub_models.scripts.notify_nightly_failures \
        --workflow-failure "Job Name" --workflow-failure-url "https://..." \
        --output-dir build/nightly-issues \
        --repository owner/repo \
        --run-url "https://github.com/..." \
        --ref-name main
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# Must match the matrix job display name defined in .github/workflows/nightly.yml
_WORKBENCH_JOB_NAME_FRAGMENT = "Verify Model Tests"

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def load_failed_jobs_json(json_path: str | None) -> dict[str, str]:
    """Load failed workbench job URLs from structured JSON file.

    Returns a dict mapping job name -> job URL.
    """
    if not json_path or not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        return json.load(f)


def categorize_failures(
    names: list[str], urls: list[str]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split workflow failures into workbench vs general."""
    workbench = []
    general = []
    for i, name in enumerate(names):
        url = urls[i] if i < len(urls) else ""
        entry = {"name": name, "url": url}
        if _WORKBENCH_JOB_NAME_FRAGMENT in name:
            workbench.append(entry)
        else:
            general.append(entry)
    return workbench, general


def render_issue_body(template_name: str, **kwargs: object) -> str:
    """Render a Jinja template from the templates directory."""
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR))
    template = env.get_template(template_name)
    return template.render(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GitHub issue bodies for nightly failures"
    )
    parser.add_argument(
        "--workflow-failure",
        action="append",
        default=[],
        dest="workflow_failures",
        help="Name of a failed workflow job (repeatable)",
    )
    parser.add_argument(
        "--workflow-failure-url",
        action="append",
        default=[],
        dest="workflow_failure_urls",
        help="URL for a failed workflow job (repeatable, matches --workflow-failure)",
    )
    parser.add_argument("--repository", required=True, help="owner/repo")
    parser.add_argument("--run-url", required=True, help="Full URL to the workflow run")
    parser.add_argument("--ref-name", required=True, help="Branch name")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write issue body files",
    )
    parser.add_argument(
        "--failed-jobs-json",
        type=str,
        default=None,
        help="Path to failed-workbench-jobs.json (structured AI Hub job URLs)",
    )
    args = parser.parse_args()

    if not args.workflow_failures:
        print("No workflow failures provided — skipping issue generation.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    workbench_failures, general_failures = categorize_failures(
        args.workflow_failures, args.workflow_failure_urls
    )

    # Load test summary if available (shared by both issues)
    test_summary = None
    summary_path = Path("build/nightly-test-results/summary.md")
    if summary_path.exists():
        test_summary = summary_path.read_text()

    # Workbench issue — only file if there's actual evidence of workbench
    # job failures (not just the verify job crashing due to infra issues).
    failed_aihub_jobs = load_failed_jobs_json(args.failed_jobs_json)
    if workbench_failures and failed_aihub_jobs:
        title = f"[QAIHM Nightly] Workbench Job Failures - {today}"
        body = render_issue_body(
            "workbench_issue.j2",
            today=today,
            failures=workbench_failures,
            failed_aihub_jobs=failed_aihub_jobs or None,
            test_summary=test_summary,
            run_url=args.run_url,
            repository=args.repository,
            ref_name=args.ref_name,
        )
        (output_dir / "workbench_title.txt").write_text(title)
        (output_dir / "workbench_issue.md").write_text(body)
        print(f"Wrote workbench issue to {output_dir}")
    elif workbench_failures:
        # The verify job failed but no workbench jobs were actually affected
        # (e.g. GitHub infra outage). Reclassify as general failures.
        general_failures.extend(workbench_failures)

    # General issue
    if general_failures:
        title = f"[QAIHM Nightly] Test Failures - {today}"
        body = render_issue_body(
            "general_issue.j2",
            today=today,
            failures=general_failures,
            test_summary=test_summary,
            run_url=args.run_url,
            repository=args.repository,
            ref_name=args.ref_name,
        )
        (output_dir / "general_title.txt").write_text(title)
        (output_dir / "general_issue.md").write_text(body)
        print(f"Wrote general issue to {output_dir}")


if __name__ == "__main__":
    main()
