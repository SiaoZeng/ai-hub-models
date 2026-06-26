# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json
import subprocess
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm
from transformers import GenerationConfig, PreTrainedTokenizerBase, set_seed

from qai_hub_models.evaluators.llm_evaluator import LLMEvaluator
from qai_hub_models.utils.base_evaluator import _DataLoader
from qai_hub_models.utils.path_helpers import QAIHM_REPO_ROOT

if TYPE_CHECKING:
    from transformers import GenerationMixin


# Default name for a pre-built grader venv. When --grader-venv isn't provided,
# a venv with this name is searched for under both the user's home dir and the
# repo root (the latter matches build_and_test.py's cwd-relative --venv
# default convention).
DEFAULT_GRADER_VENV_NAME = "qaihm-dev-grader"


@dataclass
class GeneratedResponse:
    index: int
    prompt: str
    output: str
    image_path: str | None = None


def _grader_venv_candidates(grader_venv: str | None) -> list[Path]:
    """Return the venv directories to search, in priority order.

    A user-supplied path takes priority. Otherwise look for a venv named
    ``qaihm-dev-grader`` under both the user's home dir and the repo root;
    that name lines up with the convention used by build_and_test.py.
    """
    if grader_venv:
        return [Path(grader_venv).expanduser()]
    return [
        Path.home() / DEFAULT_GRADER_VENV_NAME,
        QAIHM_REPO_ROOT / DEFAULT_GRADER_VENV_NAME,
    ]


def _resolve_grader_python(grader_venv: str | None) -> str | None:
    """Return the python executable for the grader venv, or None if absent."""
    for venv in _grader_venv_candidates(grader_venv):
        python = venv / "bin" / "python"
        if python.is_file():
            return str(python)
    return None


def _run_grader_subprocess(
    python_exe: str,
    responses_json: Path,
    output_json: Path,
) -> dict[str, Any]:
    """Invoke the grader CLI and return the parsed summary JSON.

    The grader script writes a machine-readable summary to ``output_json``.
    """
    cmd = [
        python_exe,
        "-m",
        "qai_hub_models.scripts.llm.grade_responses",
        str(responses_json),
        "--output-json",
        str(output_json),
        "--verbose",
    ]
    print(f"Running grader: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return json.loads(output_json.read_text())


def _format_grader_summary(summary: dict[str, Any]) -> str:
    counts = summary.get("counts", {})
    lines = [
        "=" * 60,
        f"Grader: {summary.get('grader_model', 'unknown')}",
        f"Responses graded: {summary.get('num_items', 0)}",
        "=" * 60,
    ]
    lines.extend(
        f"  {letter}: {counts.get(letter, 0)}" for letter in ("A", "B", "C", "D")
    )
    lines.append("")
    lines.append(
        f"Overall score: {summary.get('score_pct', 0.0):.1f}%  "
        f"({summary.get('total_points', 0)}/{summary.get('max_points', 0)} pts)"
    )
    return "\n".join(lines)


class LLMResponseEvaluator(LLMEvaluator):
    """Generate one response per prompt; persist to JSON; optionally graded.

    The contract differs from a standard accuracy-metric evaluator: ``add_batch``
    is a no-op because scoring runs in a separate Python process so it can use
    a different transformers version. The score reported by ``get_accuracy_score``
    is the grader percentage in [0, 1]; if no grader venv can be resolved, the
    responses are persisted and a ``RuntimeError`` is raised with setup guidance.

    Unlike the forward-only LLM evaluators, this one runs autoregressive
    ``generate()`` and only needs transient last-position logits, so it leaves
    the generator's logits on the model's compute device.
    """

    accumulate_logits_on_cpu = False

    def __init__(
        self,
        context_length: int,
        device: torch.device,
        tokenizer: PreTrainedTokenizerBase,
        output_dir: str | Path,
        max_new_tokens: int = 2048,
        end_tokens: set[str] | None = None,
        grader_venv: str | None = None,
        seed: int = 42,
    ) -> None:
        self.context_length = context_length
        self.device = device
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.max_new_tokens = max_new_tokens
        self.end_tokens = end_tokens or set()
        self.grader_venv = grader_venv
        self.seed = seed
        self.responses: list[GeneratedResponse] = []
        self._grader_score: float | None = None
        self._grader_report: str | None = None
        # Number of responses already written to disk; None forces a first write.
        self._persisted_count: int | None = None

    @property
    def is_distance_metric(self) -> bool:
        return False

    def reset(self) -> None:
        self.responses = []
        self._grader_score = None
        self._grader_report = None
        self._persisted_count = None

    def add_batch(self, output: Any, gt: Any) -> None:
        # Generation happens inside ``add_from_dataset``; nothing to add here.
        pass

    def _build_generation_config(self, generator: GenerationMixin) -> GenerationConfig:
        end_token_ids: list[int] = []
        for tok in self.end_tokens:
            ids = self.tokenizer.encode(tok, add_special_tokens=False)
            if len(ids) == 1:
                end_token_ids.append(ids[0])
        if self.tokenizer.eos_token_id is not None:
            end_token_ids.append(self.tokenizer.eos_token_id)
        cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            eos_token_id=end_token_ids or None,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
        )
        # HF generate() reads generation_config off the model instance.
        generator.generation_config = cfg  # type: ignore[attr-defined, unused-ignore]
        return cfg

    def _decode_response(self, output_ids: torch.Tensor, prompt_len: int | None) -> str:
        """Decode generated tokens, stripping the prompt prefix when present.

        For ``inputs_embeds`` paths HF's ``generate`` returns only the new
        tokens, so ``prompt_len`` is ``None``.
        """
        if prompt_len is None:
            return self.tokenizer.decode(output_ids, skip_special_tokens=True)  # type: ignore[return-value, unused-ignore]
        new_tokens = output_ids[prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)  # type: ignore[return-value, unused-ignore]

    def for_each_batch(
        self,
        generator: GenerationMixin,
        data: _DataLoader,
        num_samples: int | None = None,
        callback: Callable | None = None,
    ) -> None:
        set_seed(self.seed)
        self._build_generation_config(generator)

        num_samples = num_samples or len(data)
        with tqdm(total=num_samples, desc="Generating responses") as pbar:
            for i, sample in enumerate(data):
                input_ids, attention_mask, label, *rest = sample
                pixel_values = rest[0] if len(rest) > 0 else None
                image_grid_thw = rest[1] if len(rest) > 1 else None

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                if pixel_values is not None:
                    pixel_values = pixel_values.to(self.device)
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(self.device)

                if pixel_values is not None:
                    with torch.no_grad():
                        output_ids = generator.generate(  # type: ignore[operator, misc, attr-defined, unused-ignore]
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_grid_thw=image_grid_thw,
                        )
                    response = self._decode_response(output_ids[0], prompt_len=None)
                else:
                    with torch.no_grad():
                        output_ids = generator.generate(  # type: ignore[operator, misc, attr-defined, unused-ignore]
                            inputs=input_ids,
                            attention_mask=attention_mask,
                        )
                    response = self._decode_response(
                        output_ids[0], prompt_len=input_ids.shape[1]
                    )

                self.responses.append(
                    GeneratedResponse(
                        index=label.index,
                        prompt=label.prompt,
                        output=response.strip(),
                        image_path=label.image_path,
                    )
                )
                # Persist after every prompt so progress is visible on disk and
                # a crash mid-run still leaves the responses generated so far.
                self._maybe_persist_responses()
                if callback:
                    callback([input_ids, attention_mask], output_ids, label)

                pbar.update(1)
                if i + 1 >= num_samples:
                    break

    def _responses_json_path(self) -> Path:
        return self.output_dir / "responses.json"

    def _grader_summary_path(self) -> Path:
        return self.output_dir / "grader_summary.json"

    def _maybe_persist_responses(self) -> Path:
        """Write responses.json, skipping the write if nothing changed.

        ``for_each_batch`` calls this after every sample (so a crash mid-run
        still leaves a current file), and scoring calls it again. Re-writing the
        identical file each time is wasteful, so only write when new responses
        have been appended since the last write.
        """
        path = self._responses_json_path()
        if len(self.responses) == self._persisted_count:
            return path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        items = [
            {
                "idx": r.index,
                "prompt": r.prompt,
                "output": r.output,
                **({"image_path": r.image_path} if r.image_path else {}),
            }
            for r in self.responses
        ]
        path.write_text(json.dumps(items, indent=2, ensure_ascii=False))
        self._persisted_count = len(self.responses)
        return path

    def _maybe_grade(self, responses_json: Path) -> None:
        """Run the grader subprocess and cache its summary."""
        if self._grader_score is not None:
            return
        if not self.responses:
            self._grader_score = 0.0
            self._grader_report = "No responses generated."
            return

        python_exe = _resolve_grader_python(self.grader_venv)
        if python_exe is None:
            searched = ", ".join(
                str(venv) for venv in _grader_venv_candidates(self.grader_venv)
            )
            raise RuntimeError(
                f"No grader venv found (searched: {searched}). "
                f"Responses were written to {responses_json}.\n"
                f"Set one up via `python scripts/build_and_test.py "
                f"--venv {DEFAULT_GRADER_VENV_NAME} install_llm_grader_requirements`, "
                f"or pass --grader-venv <path>."
            )

        summary_path = self._grader_summary_path()
        summary = _run_grader_subprocess(python_exe, responses_json, summary_path)
        self._grader_score = float(summary.get("score_pct", 0.0)) / 100.0
        self._grader_report = _format_grader_summary(summary)

    def get_accuracy_score(self) -> float:
        """Persist responses, then run the grader.

        Returns the grader score in [0, 1]. Raises ``RuntimeError`` if no grader
        venv can be resolved (responses JSON is still written first).
        """
        responses_json = self._maybe_persist_responses()
        self._maybe_grade(responses_json)
        assert self._grader_score is not None
        return self._grader_score

    def formatted_accuracy(self) -> str:
        # Materialize the grader score; this also sets _grader_report.
        self.get_accuracy_score()
        assert self._grader_report is not None
        header = (
            f"Wrote {len(self.responses)} responses to {self._responses_json_path()}"
        )
        return textwrap.dedent(header + "\n" + self._grader_report).lstrip()
