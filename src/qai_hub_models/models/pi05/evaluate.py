# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import qai_hub as hub
import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor import PolicyProcessorPipeline
from tqdm import tqdm

from qai_hub_models.models.pi05 import Model
from qai_hub_models.models.pi05.app import Pi05App, Pi05AppConfig
from qai_hub_models.models.pi05.demo import (
    DATASET_REPO_ID,
    HF_MODEL_ID,
    _build_preprocessed_batch,
    _to_device_tree,
    compute_sequence_metrics,
)
from qai_hub_models.models.pi05.export import compile_model, link_model
from qai_hub_models.models.pi05.model import (
    NUM_ACTION_STEPS,
    Pi05ActionExpert,
    Pi05ActionExpertQuantizable,
    Pi05PaliGemmaBackbone,
    Pi05PaliGemmaBackboneBase,
    Pi05PaliGemmaBackboneQuantizable,
    Pi05PaliGemmaTokenEmbed,
    Pi05PaliGemmaVision,
    Pi05PaliGemmaVisionQuantizable,
)
from qai_hub_models.models.pi05.quantize import MIXED_PRECISION_MAP
from qai_hub_models.utils.base_model import TargetRuntime
from qai_hub_models.utils.inference import OnDeviceModel
from qai_hub_models.utils.qai_hub_helpers import assert_success_and_get_target_models


def compute_cosine_similarity(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity between predicted and GT action sequences.

    Both tensors are [B, T, D]. Aligned to min T/D, flattened per batch
    item, then averaged across the batch.
    """
    if pred.dim() != 3 or gt.dim() != 3:
        raise ValueError("pred and gt must be [B, T, D]")

    t_max = min(pred.shape[1], gt.shape[1])
    d_max = min(pred.shape[2], gt.shape[2])

    pred_flat = pred[:, :t_max, :d_max].reshape(pred.shape[0], -1)
    gt_flat = gt[:, :t_max, :d_max].reshape(gt.shape[0], -1)

    return F.cosine_similarity(pred_flat, gt_flat, dim=1).mean()


def build_gt_actions(
    dataset: LeRobotDataset,
    ep_from: int,
    ep_to: int,
    device: torch.device | str,
    seq_len: int = 50,
) -> torch.Tensor:
    """
    Build a [1, seq_len, dof] GT action sequence from an arbitrary episode.

    Parameters
    ----------
    dataset
        LeRobotDataset instance.
    ep_from
        Dataset index of the first frame of the episode.
    ep_to
        Dataset index one past the last frame of the episode.
    device
        Target device.
    seq_len
        Number of action steps.

    Returns
    -------
    torch.Tensor
        Ground-truth action tensor of shape [1, seq_len, dof].
    """
    n_frames = min(seq_len, ep_to - ep_from)
    actions = [dataset[ep_from + i]["action"] for i in range(n_frames)]
    seq = torch.stack(actions, dim=0)  # [n_frames, dof]

    if n_frames < seq_len:
        pad = seq[-1:].expand(seq_len - n_frames, -1)
        seq = torch.cat([seq, pad], dim=0)

    return seq.unsqueeze(0).to(device=device, dtype=torch.float32)


def load_episode_batch(
    dataset: LeRobotDataset,
    ep_start_idx: int,
    pi05_config: PI05Config,
    device: str = "cpu",
) -> tuple[dict[str, Any], PolicyProcessorPipeline]:
    """Load and preprocess the first frame of an episode for Pi05 inference."""
    raw_sample = dataset[ep_start_idx]

    raw_batch: dict[str, Any] = {}
    for k, v in raw_sample.items():
        if isinstance(v, torch.Tensor):
            raw_batch[k] = v.unsqueeze(0)
        elif isinstance(v, str):
            raw_batch[k] = [v]
        else:
            raw_batch[k] = v

    batch, postprocessor = _build_preprocessed_batch(
        cfg=pi05_config,
        raw_batch=raw_batch,
        batch_size=1,
        dataset_stats=dataset.meta.stats,
    )
    batch = _to_device_tree(batch, device)
    return batch, postprocessor


_COMPONENT_SPEC: dict[str, tuple[list[str], list[str]]] = {
    "vision_encoder": (
        list(Pi05PaliGemmaVision.get_input_spec().keys()),
        Pi05PaliGemmaVision.get_output_names(),
    ),
    "token_emb": (
        list(Pi05PaliGemmaTokenEmbed.get_input_spec().keys()),
        Pi05PaliGemmaTokenEmbed.get_output_names(),
    ),
    "backbone": (
        list(Pi05PaliGemmaBackbone.get_input_spec().keys()),
        Pi05PaliGemmaBackboneBase.get_output_names((0, 18)),
    ),
    "action_expert": (
        list(Pi05ActionExpert.get_input_spec().keys()),
        Pi05ActionExpert.get_output_names(),
    ),
}


def compile_for_device(
    device_name: str,
    target_runtime: TargetRuntime = TargetRuntime.QNN_CONTEXT_BINARY,
) -> dict[str, hub.Model]:
    """Compile the 4 on-device Pi05 components and return hub.Model handles."""
    hub_device = hub.Device(device_name)
    model = Model.from_pretrained()
    output_path = Path.cwd() / "export_assets"

    print("Compiling Pi05 components for on-device inference...")
    compile_jobs = compile_model(
        model,
        "pi05_eval",
        hub_device,
        target_runtime,
        output_path=output_path,
        components=list(_COMPONENT_SPEC.keys()),
    )
    compiled_models = assert_success_and_get_target_models(compile_jobs)

    if target_runtime.uses_hub_link:
        link_jobs = link_model(
            compiled_models,
            hub_device,
            "pi05_eval",
            model,
            target_runtime,
        )
        return assert_success_and_get_target_models(link_jobs)

    return compiled_models


def evaluate_on_device(
    fp_app: Pi05App,
    dataset: LeRobotDataset,
    policy: PI05Policy,
    target_models: dict[str, hub.Model],
    device_name: str,
    num_samples: int,
    seed: int,
    ep_from: list[int],
    torch_device: str = "cpu",
) -> None:
    """Run batched on-device evaluation comparing device outputs to FP reference."""
    hub_device = hub.Device(device_name)

    od_components: dict[str, OnDeviceModel] = {}
    for comp_name, (input_names, output_names) in _COMPONENT_SPEC.items():
        od_components[comp_name] = OnDeviceModel(
            model=target_models[comp_name],
            input_names=input_names,
            device=hub_device,
            output_names=output_names,
        )

    app_cfg = Pi05AppConfig.from_policy(policy)
    od_app = Pi05App(config=app_cfg, **od_components)

    print(f"Phase 1: Preprocessing {num_samples} episodes...")
    all_batches: list[dict[str, Any]] = []
    all_postprocessors: list[PolicyProcessorPipeline] = []
    for ep_idx in tqdm(range(num_samples), desc="Loading episodes"):
        batch, postprocessor = load_episode_batch(
            dataset, int(ep_from[ep_idx]), policy.config, torch_device
        )
        all_batches.append(batch)
        all_postprocessors.append(postprocessor)

    mega_batch: dict[str, Any] = {}
    for key in all_batches[0]:
        vals = [b[key] for b in all_batches]
        if isinstance(vals[0], torch.Tensor):
            mega_batch[key] = torch.cat(vals, dim=0)
        elif isinstance(vals[0], list):
            mega_batch[key] = [item for sublist in vals for item in sublist]
        else:
            mega_batch[key] = vals

    all_noise = []
    for ep_idx in range(num_samples):
        torch.manual_seed(seed + ep_idx)
        all_noise.append(
            torch.normal(0.0, 1.0, size=(1, NUM_ACTION_STEPS, 32), dtype=torch.float32)
        )
    mega_noise = torch.cat(all_noise, dim=0)

    print("Phase 2: Running on-device inference (13 hub jobs)...")
    od_raw = od_app.predict_action_chunk(
        mega_batch, noise=mega_noise, truncate_action_by_dof=False
    )

    print("Phase 3: Computing FP reference predictions...")
    fp_predictions: list[torch.Tensor] = []
    for ep_idx in tqdm(range(num_samples), desc="FP inference"):
        torch.manual_seed(seed + ep_idx)
        fp_raw = fp_app.predict_action_chunk(batch=all_batches[ep_idx], noise=None)
        fp_predictions.append(all_postprocessors[ep_idx](fp_raw).cpu())

    print("Phase 4: Computing metrics...")
    action_dof = fp_app.action_dof
    mse_sum = 0.0
    sqnr_sum = 0.0
    cos_sum = 0.0

    for ep_idx in range(num_samples):
        pred_raw = od_raw[ep_idx : ep_idx + 1, :, :action_dof]
        pred = all_postprocessors[ep_idx](pred_raw).cpu()
        reference = fp_predictions[ep_idx]

        mse, _rmse, sqnr = compute_sequence_metrics(pred, reference)
        cos_sim = compute_cosine_similarity(pred, reference)
        mse_sum += float(mse)
        sqnr_sum += float(sqnr)
        cos_sum += float(cos_sim)

    mean_mse = mse_sum / num_samples
    mean_sqnr = sqnr_sum / num_samples
    mean_cos = cos_sum / num_samples

    print(f"\n===== Aggregate ({num_samples} episodes, mode=on-device) =====")
    print(f"  Mean MSE:    {mean_mse:.6f}")
    print(f"  Mean SQNR:   {mean_sqnr:.1f} dB")
    print(f"  Mean CosSim: {mean_cos * 100:.1f}%")
    print("=============================================")


def main(is_test: bool = False) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Pi05 model on LIBERO dataset (FP or quantsim)."
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["fp", "quantsim", "on-device"],
        default="quantsim",
        help="'fp': FP predictions vs ground truth. "
        "'quantsim': quantsim predictions vs FP predictions. "
        "'on-device': on-device predictions vs FP predictions. "
        "Default: quantsim.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Quantsim checkpoint directory (only used with --eval-mode=quantsim). "
        "Default: S3-cached checkpoint (pass local path to override).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of episodes to evaluate. Default: 100. Use --num-samples=0 for all.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="Dragonwing IQ-9075 EVK",
        help="AI Hub device name (only for --eval-mode=on-device). "
        "Default: 'Dragonwing IQ-9075 EVK'.",
    )
    parser.add_argument(
        "--hub-model-ids",
        type=str,
        default=None,
        help="JSON string mapping component names to hub model IDs "
        "(only for --eval-mode=on-device). "
        'Example: \'{"vision_encoder":"m1","token_emb":"m2",'
        '"backbone":"m3","action_expert":"m4"}\'',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=50)
    args = parser.parse_args([] if is_test else None)

    seed = 1234 if is_test else args.seed
    num_samples = 1 if is_test else args.num_samples
    eval_mode = "fp" if is_test else args.eval_mode
    checkpoint = args.checkpoint or "DEFAULT"
    seq_len = args.seq_len

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = PI05Policy.from_pretrained(HF_MODEL_ID).to(device).eval()
    dataset = LeRobotDataset(DATASET_REPO_ID)

    episodes = dataset.meta.episodes
    ep_from = episodes["dataset_from_index"]
    ep_to = episodes["dataset_to_index"]
    n_episodes = len(ep_from)

    if num_samples is None or num_samples <= 0:
        num_samples = n_episodes
    num_samples = min(num_samples, n_episodes)
    print(
        f"Using {num_samples}/{n_episodes} episodes. "
        f"Use --num-samples=0 for the full dataset."
    )

    app_cfg = Pi05AppConfig.from_policy(policy)
    fp_components = {
        "vision_encoder": Pi05PaliGemmaVision(policy),
        "token_emb": Pi05PaliGemmaTokenEmbed(policy),
        "action_expert": Pi05ActionExpert(policy),
        "backbone": Pi05PaliGemmaBackbone(policy),
    }
    fp_app = Pi05App(config=app_cfg, **fp_components).to(device).eval()

    if eval_mode == "on-device":
        if args.hub_model_ids:
            hub_model_ids = json.loads(args.hub_model_ids)
            for comp in _COMPONENT_SPEC:
                if comp not in hub_model_ids:
                    parser.error(f"--hub-model-ids must include '{comp}'")
            target_models = {
                comp: hub.get_model(hub_model_ids[comp]) for comp in _COMPONENT_SPEC
            }
        else:
            target_models = compile_for_device(args.device)
        evaluate_on_device(
            fp_app=fp_app,
            dataset=dataset,
            policy=policy,
            target_models=target_models,
            device_name=args.device,
            num_samples=num_samples,
            seed=seed,
            ep_from=ep_from,
            torch_device=device,
        )
        return

    quantsim_app = None
    if eval_mode == "quantsim":
        qs_components: dict[str, torch.nn.Module] = dict(fp_components)

        vit_prec = MIXED_PRECISION_MAP["vision_encoder"]
        qs_vit = Pi05PaliGemmaVisionQuantizable.from_pretrained(
            checkpoint=checkpoint,
            host_device=device,
            precision=vit_prec,
        )
        if qs_vit.quant_sim is not None:
            providers = qs_vit.quant_sim.session.get_providers()
            print(
                f"Quantsim vision_encoder (precision={vit_prec}) providers: {providers}"
            )
        else:
            print(f"vision_encoder (precision={vit_prec}) loaded in bundle-only mode")
        qs_components["vision_encoder"] = qs_vit.to(device)

        bb_prec = MIXED_PRECISION_MAP["backbone"]
        qs_backbone = Pi05PaliGemmaBackboneQuantizable.from_pretrained(
            checkpoint=checkpoint,
            host_device=device,
            precision=bb_prec,
        )
        if qs_backbone.quant_sim is not None:
            providers = qs_backbone.quant_sim.session.get_providers()
            print(f"Quantsim backbone (precision={bb_prec}) providers: {providers}")
        else:
            print(f"backbone (precision={bb_prec}) loaded in bundle-only mode")
        qs_components["backbone"] = qs_backbone.to(device)

        ae_prec = MIXED_PRECISION_MAP["action_expert"]
        qs_action_expert = Pi05ActionExpertQuantizable.from_pretrained(
            checkpoint=checkpoint,
            host_device=device,
            precision=ae_prec,
        )
        if qs_action_expert.quant_sim is not None:
            providers = qs_action_expert.quant_sim.session.get_providers()
            print(
                f"Quantsim action_expert (precision={ae_prec}) providers: {providers}"
            )
        else:
            print(f"action_expert (precision={ae_prec}) loaded in bundle-only mode")
        qs_components["action_expert"] = qs_action_expert.to(device)

        quantsim_app = (
            Pi05App(
                config=app_cfg,
                **qs_components,
            )
            .to(device)
            .eval()
        )

    if eval_mode == "fp":
        print("Evaluating: FP predictions vs ground truth")
    else:
        print(
            "Evaluating: quantsim predictions vs FP predictions (measuring quantization error)"
        )

    mse_sum = 0.0
    sqnr_sum = 0.0
    cos_sum = 0.0
    last_print_time = time.monotonic()

    pbar = tqdm(range(num_samples), desc="Evaluating", unit="ep")
    for ep_idx in pbar:
        batch, postprocessor = load_episode_batch(
            dataset, int(ep_from[ep_idx]), policy.config, device
        )

        if eval_mode == "fp":
            torch.manual_seed(seed + ep_idx)
            pred_raw = fp_app.predict_action_chunk(batch=batch, noise=None)
            pred = postprocessor(pred_raw).to(device)
            reference = build_gt_actions(
                dataset, int(ep_from[ep_idx]), int(ep_to[ep_idx]), device, seq_len
            )
        else:
            # Run both FP and quantsim with the same noise
            assert quantsim_app is not None
            torch.manual_seed(seed + ep_idx)
            fp_raw = fp_app.predict_action_chunk(batch=batch, noise=None)
            reference = postprocessor(fp_raw).to(device)

            torch.manual_seed(seed + ep_idx)
            qs_raw = quantsim_app.predict_action_chunk(batch=batch, noise=None)
            pred = postprocessor(qs_raw).to(device)

        mse, _rmse, sqnr = compute_sequence_metrics(pred, reference)
        cos_sim = compute_cosine_similarity(pred, reference)

        mse_sum += float(mse)
        sqnr_sum += float(sqnr)
        cos_sum += float(cos_sim)
        n = ep_idx + 1

        avg_mse = mse_sum / n
        avg_sqnr = sqnr_sum / n
        avg_cos = cos_sum / n

        pbar.set_postfix_str(
            f"MSE={avg_mse:.6f}  SQNR={avg_sqnr:.1f}dB  CosSim={avg_cos * 100:.1f}%"
        )

        now = time.monotonic()
        if now - last_print_time >= 60:
            tqdm.write(
                f"[ep {ep_idx:>4d}] running avg: "
                f"MSE={avg_mse:.6f}  "
                f"SQNR={avg_sqnr:.1f} dB  "
                f"CosSim={avg_cos * 100:.1f}%"
            )
            last_print_time = now

    mean_mse = mse_sum / num_samples
    mean_sqnr = sqnr_sum / num_samples
    mean_cos = cos_sum / num_samples

    print(f"\n===== Aggregate ({num_samples} episodes, mode={eval_mode}) =====")
    print(f"  Mean MSE:    {mean_mse:.6f}")
    print(f"  Mean SQNR:   {mean_sqnr:.1f} dB")
    print(f"  Mean CosSim: {mean_cos * 100:.1f}%")
    print("=============================================")

    if is_test:
        assert mean_mse < 0.001, f"MSE {mean_mse} >= 0.001"
        assert mean_sqnr > 25, f"SQNR {mean_sqnr} <= 25"
        assert mean_cos > 0.95, f"CosSim {mean_cos} <= 0.95"


if __name__ == "__main__":
    main()
