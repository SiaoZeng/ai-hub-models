# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from functools import partial
from typing import Any

from qai_hub_models import Precision
from qai_hub_models.models._shared.llm.app import ChatApp as App
from qai_hub_models.models._shared.llm.model import (
    LLM_QNN,
    LLM_AIMETOnnx,
    LLMBase,
    LLMDynamicBase,
    get_tokenizer,
)
from qai_hub_models.utils.args import add_input_spec_args, get_model_cli_parser
from qai_hub_models.utils.checkpoint import (
    CheckpointSpec,
    CheckpointType,
)
from qai_hub_models.utils.huggingface import has_model_access

# Max output tokens to generate
# You can override this with cli argument.
MAX_OUTPUT_TOKENS = 1000


def llm_chat_demo(
    model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    qnn_model_cls: type[LLM_QNN],
    model_id: str,
    end_tokens: set[str],
    hf_repo_name: str,
    hf_repo_url: str,
    supported_precisions: list[Precision],
    default_prompt: str | None = None,
    raw: bool = False,
    test_checkpoint: CheckpointSpec | None = None,
    supports_thinking: bool = False,
    # VLM parameters (optional)
    vision_encoder_cls: Any | None = None,
    hidden_size: int | None = None,
) -> None:
    """Shared Chat Demo App to generate output for provided input prompt"""
    # Demo parameters
    parser = get_model_cli_parser(
        model_cls,
        suppress_help_arguments=["--host-device", "--fp-model", "--precision"],
    )
    parser = add_input_spec_args(model_cls, parser)
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="input prompt.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="input prompt from file path.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="If specified, will assume prompt contains systems tags and will not be added automatically.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=MAX_OUTPUT_TOKENS,
        help="max output tokens to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed.",
    )
    # VLM image argument (only shown if vision_encoder_cls is provided)
    if vision_encoder_cls is not None:
        parser.add_argument(
            "--image",
            type=str,
            nargs="+",
            default=None,
            help="Path(s) to input image(s). Pass multiple paths for multi-image prompts.",
        )
        parser.add_argument(
            "--image-height",
            type=int,
            default=None,
            help="VEG image height (must be divisible by patch_size * spatial_merge_size).",
        )
        parser.add_argument(
            "--image-width",
            type=int,
            default=None,
            help="VEG image width (must be divisible by patch_size * spatial_merge_size).",
        )
    if supports_thinking:
        parser.add_argument(
            "--thinking",
            action="store_true",
            dest="thinking",
            default=True,
            help="Enable thinking mode (default).",
        )
        parser.add_argument(
            "--no-thinking",
            action="store_false",
            dest="thinking",
            help="Disable thinking mode by adding empty thinking tags.",
        )

    args = parser.parse_args([] if test_checkpoint is not None else None)
    checkpoint = args.checkpoint if test_checkpoint is None else test_checkpoint
    max_output_tokens = args.max_output_tokens if test_checkpoint is None else 1000
    if args.prompt is not None and args.prompt_file is not None:
        raise ValueError("Must specify one of --prompt or --prompt-file")
    if args.prompt_file is not None:
        with open(args.prompt_file) as f:
            prompt = f.read()
    elif args.prompt:
        prompt = args.prompt
    elif default_prompt is not None:
        prompt = default_prompt
    else:
        prompt = fp_model_cls.default_user_prompt

    # Make sure that we can pass "\n" (0x0A) as part of the prompt, since that
    # is often a common feature of prompt formats. If this gets interpreted as
    # "\\n" (0x5C 0x6E), the LLM can react poorly (quantized models have been
    # observed to be particularly sensitive to this).
    prompt = prompt.replace("\\n", "\n")

    assert checkpoint is not None
    checkpoint_type = CheckpointType.from_checkpoint(checkpoint)
    is_default = isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT")
    if not is_default:
        tokenizer = get_tokenizer(checkpoint)
    else:
        has_model_access(hf_repo_name, hf_repo_url)
        tokenizer = get_tokenizer(hf_repo_name)

    # Build the prompt formatting function
    if args.raw or raw:

        def preprocess_prompt_fn(
            user_input_prompt: str = "",
            system_context_prompt: str = "",
        ) -> str:
            return user_input_prompt
    elif supports_thinking:
        preprocess_prompt_fn = partial(
            fp_model_cls.get_input_prompt_with_tags,
            tokenizer=tokenizer,
            enable_thinking=args.thinking,
        )
    else:
        preprocess_prompt_fn = partial(
            fp_model_cls.get_input_prompt_with_tags, tokenizer=tokenizer
        )

    # Get image path if VLM
    image_path = (
        getattr(args, "image", None) if vision_encoder_cls is not None else None
    )

    if test_checkpoint is None:
        print(f"\n{'-' * 85}")
        print(f"** Generating response via {model_id} **")
        if checkpoint_type == CheckpointType.GENIE_BUNDLE:
            print("Variant: ON-DEVICE (QNN)")
            print("    This runs on the target hardware.")
        elif checkpoint_type.is_aimet_onnx():
            print("Variant: QUANTIZED (AIMET-ONNX)")
            print("    This aims to replicate on-device accuracy through simulation.")
        else:
            print("Variant: FLOATING POINT (PyTorch)")
            print("    This runs the original unquantized model for baseline purposes.")
        print()
        print("Prompt:", prompt)
        if vision_encoder_cls is not None:
            print("Image:", image_path if image_path else "(none - text only)")
        print("Raw (prompt will be passed in unchanged):", args.raw)
        if supports_thinking:
            print("Thinking mode:", "enabled" if args.thinking else "disabled")
        print("Max number of output tokens to generate:", args.max_output_tokens)
        print()
        print(f"{'-' * 85}\n")

    extra = {}

    final_model_cls: type[LLMBase | LLM_AIMETOnnx | LLM_QNN]

    if checkpoint_type == CheckpointType.GENIE_BUNDLE:
        final_model_cls = qnn_model_cls

    elif checkpoint_type.is_aimet_onnx():
        if is_default and checkpoint != "DEFAULT_UNQUANTIZED":
            fp_kwargs: dict[str, Any] = {}
            if not issubclass(fp_model_cls, LLMDynamicBase):
                fp_kwargs["sequence_length"] = args.sequence_length
                fp_kwargs["context_length"] = args.context_length
            extra["fp_model"] = fp_model_cls.from_pretrained(**fp_kwargs)
        final_model_cls = model_cls
    else:
        final_model_cls = fp_model_cls

    # Collect VLM image size override if provided
    vlm_image_size = None
    if vision_encoder_cls is not None:
        img_h = getattr(args, "image_height", None)
        img_w = getattr(args, "image_width", None)
        if img_h is not None and img_w is not None:
            vlm_image_size = (img_w, img_h)

    app = App(
        final_model_cls,
        get_input_prompt_with_tags=preprocess_prompt_fn,
        tokenizer=tokenizer,
        end_tokens=end_tokens,
        seed=args.seed,
        # VLM parameters
        vision_encoder_cls=vision_encoder_cls,
        hf_repo_name=hf_repo_name if vision_encoder_cls is not None else None,
        hidden_size=hidden_size,
        vlm_image_size=vlm_image_size,
    )

    app.generate_output_prompt(
        prompt,
        context_length=args.context_length,
        max_output_tokens=max_output_tokens,
        checkpoint=checkpoint,
        model_from_pretrained_extra=extra,
        image_path=image_path,
        sequence_length=args.sequence_length,
    )
