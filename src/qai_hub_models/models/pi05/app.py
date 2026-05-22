# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models.models.pi05.model import (
    DEFAULT_CHECKPOINT,
    NUM_ACTION_STEPS,
    NUM_CAMERAS,
    Pi05ActionExpert,
    Pi05PaliGemmaBackbone,
    Pi05PaliGemmaTokenEmbed,
    Pi05PaliGemmaVision,
    load_checkpoint,
)
from qai_hub_models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.base_model import PretrainedCollectionModel
from qai_hub_models.utils.image_processing import resize_pad
from qai_hub_models.utils.inference import OnDeviceModel
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries


@dataclass
class Pi05AppConfig:
    """
    Lightweight configuration for Pi05App. This mirrors the parts of the
    policy/model config that Pi05App needs, allowing construction without
    passing a full policy object.
    """

    # Number of action steps used by the model.
    # Default value aligns with common PI0 configuration.
    n_action_steps: int = 50

    # Maximum action dimension (e.g., 32 for flow model config).
    max_action_dim: int = 32

    # List of image keys corresponding to visual input features.
    image_keys: list[str] | None = None

    # Actual degree of freedom, not the max action dim (e.g., 7 or 9).
    action_dof: int = 0

    # If True, use the RTC-unrolled expert for inference. When enabled,
    # predict_action_chunk/sample_action require prev_actions.
    use_rtc: bool = False

    @staticmethod
    def from_policy(policy: PI05Policy, use_rtc: bool = False) -> Pi05AppConfig:
        """
        Build Pi05AppConfig from an existing policy. This inspects the
        policy to extract only the fields required by Pi05App.
        """
        cfg_model = policy.model.config
        image_keys = [
            k
            for k, v in cfg_model.input_features.items()
            if v.type == FeatureType.VISUAL and "empty" not in k
        ]
        action_dof = policy.config.output_features["action"].shape[0]
        return Pi05AppConfig(
            n_action_steps=cfg_model.n_action_steps,
            max_action_dim=cfg_model.max_action_dim,
            image_keys=image_keys,
            action_dof=action_dof,
            use_rtc=use_rtc,
        )


def _unbatch(tensor: torch.Tensor) -> list[torch.Tensor | np.ndarray]:
    """Split [B, ...] into list of B tensors each [1, ...]."""
    return list(torch.unbind(tensor.unsqueeze(1)))


class BatchedOnDeviceModel:
    """
    Wraps OnDeviceModel to present a torch-module-like interface.

    Accepts batched tensors as positional args (like a torch module),
    internally unbatches them into per-sample lists for OnDeviceModel,
    and returns the result as a tensor or tuple of tensors.
    """

    def __init__(self, model: OnDeviceModel) -> None:
        self.model = model

    def __call__(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return self.model(*[_unbatch(t) for t in args])


def resize_and_normalize(image: torch.Tensor) -> torch.Tensor:
    """Resize with padding to 224x224 and normalize to [-1, 1]."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image, _, _ = resize_pad(
        image, (224, 224), vertical_float="top", horizontal_float="left"
    )
    return image * 2.0 - 1.0


class Pi05App(torch.nn.Module):
    """
    Assemble Pi05Collection parts to reproduce the core computation of
    PI05Policy.forward (i.e., FlowMatching forward). This class expects a
    batch coming from LeRobot's dataset and uses the provided policy's
    prepare_* utilities to match PI05Policy.forward preprocessing.

    Expected keys in batch (as observed):
      - "observation.images.*": torch.Tensor [B, 3, H, W]
      - "action" (ACTION): torch.Tensor [B, T, D]
      - "observation.state" (OBS_STATE): torch.Tensor [B, S]
      - "action_is_pad": torch.BoolTensor [B, T] (optional)
      - "task": list[str] (natural language instruction)

    Required forward inputs:
      - noise: torch.Tensor [B, T, Dcfg] produced by the caller
      - time: torch.Tensor [B]

    Forward returns a tuple (loss, loss_dict) where loss is a scalar tensor
    suitable for backward and loss_dict contains "losses_after_forward" with
    shape [B, Tcfg, Dcfg] where Dcfg == model.config.max_action_dim.
    """

    def __init__(
        self,
        config: Pi05AppConfig,
        vision_encoder: ExecutableModelProtocol,
        token_emb: ExecutableModelProtocol,
        action_expert: ExecutableModelProtocol,
        backbone: ExecutableModelProtocol,
    ) -> None:
        """
        Initialize Pi05App with model components.

        Parameters
        ----------
        config
            Pi05AppConfig containing model configuration.
        vision_encoder
            Vision encoder component.
        token_emb
            Token embedding component.
        action_expert
            Action expert component for denoising.
        backbone
            Full backbone (layers 0-18).
        """
        super().__init__()

        # When components are OnDeviceModel instances, wrap them so that
        # call sites can pass batched tensors directly (like a torch module).
        self._on_device = isinstance(vision_encoder, OnDeviceModel)
        self.vision_encoder = vision_encoder
        self.token_emb = (
            BatchedOnDeviceModel(token_emb)
            if isinstance(token_emb, OnDeviceModel)
            else token_emb
        )
        self.action_expert = (
            BatchedOnDeviceModel(action_expert)
            if isinstance(action_expert, OnDeviceModel)
            else action_expert
        )
        self.backbone = (
            BatchedOnDeviceModel(backbone)
            if isinstance(backbone, OnDeviceModel)
            else backbone
        )

        # Cache a few config bits from the provided flow config. This removes
        # the hard dependency on a policy object while keeping the exact
        # fields Pi05App uses.
        self.n_action_steps: int = config.n_action_steps
        self.max_action_dim: int = config.max_action_dim
        self.image_keys = config.image_keys
        # Actual degree of freedom, not the max action dim (32).
        self.action_dof = config.action_dof
        # Whether to use RTC-unrolled expert during inference.
        self.use_rtc: bool = bool(config.use_rtc)

    def _resize_and_normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"[B,C,H,W] expected, got {image.shape}")
        return resize_and_normalize(image)

    def populate_prefix(
        self,
        img_ls: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
    ]:
        """
        Run vision encoders, token embedding, and PaliGemma backbone chunks
        to build the KV caches and auxiliary tensors required by the
        expert.

        Parameters
        ----------
        img_ls
            List of tensors, each [B, 3, H, W]. Supports a variable
            number of images. The first image is mapped to the primary
            stream; any remaining images are concatenated to the secondary
            stream sequence.
        lang_tokens
            Tensor of token ids [B, Ltok].
        lang_mask
            Bool tensor of attention mask [B, Ltok].

        Returns
        -------
        suffix_sin : torch.Tensor
            Float tensor for RoPE sine embeddings used for
            suffix (actions); shape [B, Lsuffix, Hd].
        suffix_cos : torch.Tensor
            Float tensor for RoPE cosine embeddings used for
            suffix (actions); shape [B, Lsuffix, Hd].
        k_all : list[torch.Tensor]
            list of key-cache tensors for layers 0..17. Each item
            has shape [B, n_heads, Lprefix, head_dim] or per-impl eqv.
        v_all : list[torch.Tensor]
            list of value-cache tensors for layers 0..17. Each item
            has shape [B, n_heads, Lprefix, head_dim] or per-impl eqv.
        full_att_4d : torch.Tensor
            float mask [B, 1, Ls, Lp+Ls] to be used additively
            on attention logits (0 allowed, -1e4 blocked).
        """
        if len(img_ls) == 0:
            raise ValueError("populate_prefix requires at least one image.")

        # Resize to 224x224 with padding and normalize to [-1, 1].
        proc_imgs = [self._resize_and_normalize_image(x) for x in img_ls]

        # Vision encodings (each returns [B, S_img, D]).
        if self._on_device:
            bsize = proc_imgs[0].shape[0]
            num_cams = len(proc_imgs)
            all_imgs: list[torch.Tensor | np.ndarray] = []
            for x in proc_imgs:
                all_imgs.extend(_unbatch(x))
            combined = self.vision_encoder(all_imgs)
            assert isinstance(combined, torch.Tensor)
            img_embeds = [
                combined[i * bsize : (i + 1) * bsize] for i in range(num_cams)
            ]
        else:
            img_embeds = [self.vision_encoder(x) for x in proc_imgs]

        # Token embedding packs images + language and produces prefix
        # embeddings/masks and RoPE tensors.
        # On-device path needs padding to NUM_CAMERAS (fixed input spec).
        if self._on_device:
            padded_embeds: list[torch.Tensor] = list(img_embeds)
            if len(padded_embeds) < NUM_CAMERAS:
                base = img_embeds[0]
                padded_embeds.extend(
                    torch.zeros_like(base)
                    for _ in range(NUM_CAMERAS - len(padded_embeds))
                )
            te_out = self.token_emb(
                lang_tokens,
                lang_mask.to(dtype=torch.float32),
                *padded_embeds,
            )
        else:
            te_out = self.token_emb(
                lang_tokens,
                lang_mask.to(dtype=torch.float32),
                *img_embeds,
            )
        assert isinstance(te_out, tuple)
        (
            prefix_emb,
            prefix_att_2d,
            prefix_sin,
            prefix_cos,
            suffix_sin,
            suffix_cos,
            full_att_4d,
        ) = te_out

        # Run PaliGemma backbone to fill per-layer KV caches for the
        # prefix.
        rest_full = self.backbone(prefix_emb, prefix_att_2d, prefix_sin, prefix_cos)
        n_full = len(rest_full) // 2
        k_all = list(rest_full[:n_full])
        v_all = list(rest_full[n_full:])

        return suffix_sin, suffix_cos, k_all, v_all, full_att_4d

    def denoise_step(
        self,
        k_all: list[torch.Tensor],
        v_all: list[torch.Tensor],
        suffix_sin: torch.Tensor,
        suffix_cos: torch.Tensor,
        x_t: torch.Tensor,
        time_step: torch.Tensor,
        full_att_4d: torch.Tensor,
        prev_chunk: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply one denoising step using the cached prefix and the action
        expert, mirroring PI0FlowMatching.denoise_step behavior.

        Parameters
        ----------
        k_all
            List of key-cache tensors for layers 0..17. Each item
            typically has shape [B, n_heads, Lprefix, head_dim].
        v_all
            List of value-cache tensors for layers 0..17. Each item
            typically has shape [B, n_heads, Lprefix, head_dim].
        suffix_sin
            Float tensor RoPE sine embedding for suffix with
            shape [B, Lsuffix, Hd].
        suffix_cos
            Float tensor RoPE cosine embedding for suffix with
            shape [B, Lsuffix, Hd].
        x_t
            Float tensor of noisy actions [B, Tcfg, Dcfg].
        time_step
            Float tensor [B] with values in [0, 1].
        full_att_4d
            Float additive mask [B,1,Ls,Lp+Ls] for attention.
        prev_chunk
            When use_rtc is True, the previous action chunk
            with shape [B, Tcfg, Dcfg]. Ignored otherwise.

        Returns
        -------
        updated_actions : torch.Tensor
            Tensor [B, Tcfg, Dcfg] of updated actions x_{t+dt}.
        """
        if self._on_device:
            # Positional args in input_spec order: full_att_4d, rope_emb_sin,
            # rope_emb_cos, x_t, time_step, key_cache_l0..17, value_cache_l0..17
            result = self.action_expert(
                full_att_4d,
                suffix_sin,
                suffix_cos,
                x_t.to(torch.float32),
                time_step,
                *[k.to(torch.float32) for k in k_all],
                *[v.to(torch.float32) for v in v_all],
            )
            assert isinstance(result, torch.Tensor)
            return result

        action_kwargs: dict[str, torch.Tensor] = {
            "rope_emb_sin": suffix_sin,
            "rope_emb_cos": suffix_cos,
            "x_t": x_t.to(torch.float32),
            "full_att_4d": full_att_4d,
            "time_step": time_step,
        }
        for i, k in enumerate(k_all):
            action_kwargs[f"key_cache_l{i}"] = k.to(torch.float32)
        for i, v in enumerate(v_all):
            action_kwargs[f"value_cache_l{i}"] = v.to(torch.float32)

        if self.use_rtc:
            if prev_chunk is None:
                raise ValueError("prev_chunk must be provided when use_rtc is True.")
            return self.action_expert(  # type: ignore[call-arg, return-value]
                prev_chunk=prev_chunk.to(torch.float32),
                **action_kwargs,
            )

        # Expert returns x_{t+dt} after an internal Euler step.
        return self.action_expert(**action_kwargs)  # type: ignore[return-value]

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, torch.Tensor | Any],
        noise: torch.Tensor | None = None,
        num_steps: int = 10,
        prev_actions: torch.Tensor | None = None,
        truncate_action_by_dof: bool = True,
    ) -> torch.Tensor:
        lang_tokens = batch["observation.language.tokens"]  # [B, 200]
        assert self.image_keys is not None
        img_ls = [batch[key] for key in self.image_keys]  # [B, 3, 224, 224]
        # [B, 200] (float)
        lang_mask = batch["observation.language.attention_mask"].to(torch.float32)

        if self.use_rtc and prev_actions is None:
            raise ValueError("prev_actions is required when use_rtc is True.")

        # Sample actions using the model (no robot state needed in Pi05).
        actions = self.sample_action(
            img_ls=img_ls,
            lang_tokens=lang_tokens,
            lang_mask=lang_mask,
            noise=noise,
            num_steps=num_steps,
            prev_actions=prev_actions,
        )

        # Truncate to the policy-configured output action dim.
        if not truncate_action_by_dof:
            return actions
        return actions[:, :, : self.action_dof]

    @torch.no_grad()
    def sample_action(
        self,
        img_ls: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_mask: torch.Tensor,
        noise: torch.Tensor | None = None,
        num_steps: int = 10,
        prev_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform inference by running prefix population once and then
        applying Euler updates for a fixed number of steps, similar to
        PI0FlowMatching.sample_actions.

        Parameters
        ----------
        img_ls
            List of [B, 3, H, W] images.
        lang_tokens
            Tensor [B, Ltok] language tokens.
        lang_mask
            Bool tensor [B, Ltok] language mask.
        noise
            Optional float tensor [B, Tcfg, Dcfg]. If None, a
            standard normal tensor is sampled.
        num_steps
            Integer Euler steps. Default is 10.
        prev_actions
            When use_rtc is True, provide the previous
            chunk [B, Tcfg, Dcfg] whose first entry corresponds to the
            next action to execute.

        Returns
        -------
        denoised_actions : torch.Tensor
            Float tensor [B, Tcfg, Dcfg] of denoised actions that are
            already denormalized back to the action space.
        """
        if len(img_ls) == 0:
            raise ValueError("sample_action requires at least one image.")

        if self.use_rtc and prev_actions is None:
            raise ValueError("prev_actions is required when use_rtc is True.")

        device = img_ls[0].device
        bsize = lang_tokens.shape[0]

        (
            suffix_sin,
            suffix_cos,
            k_all,
            v_all,
            full_att_4d,
        ) = self.populate_prefix(
            img_ls=img_ls,
            lang_tokens=lang_tokens,
            lang_mask=lang_mask.to(torch.float32),
        )

        # Initialize noise if not provided.
        if noise is None:
            actions_shape = (
                bsize,
                self.n_action_steps,
                self.max_action_dim,
            )
            noise = torch.normal(
                mean=0.0,
                std=1.0,
                size=actions_shape,
                dtype=torch.float32,
                device=device,
            )

        # Euler integration from t=1 down to ~0.
        dt = torch.tensor(
            -1.0 / float(num_steps),
            dtype=torch.float32,
            device=device,
        )
        x_t = noise
        t_cur = torch.tensor(1.0, dtype=torch.float32, device=device)
        while t_cur >= -dt / 2:
            time_b = t_cur.expand(bsize)  # [B]
            # Expert returns x_{t+dt}; advance t here only.
            x_t = self.denoise_step(
                k_all=k_all,
                v_all=v_all,
                suffix_sin=suffix_sin,
                suffix_cos=suffix_cos,
                x_t=x_t,
                time_step=time_b,
                full_att_4d=full_att_4d,
                prev_chunk=prev_actions,
            )
            t_cur = t_cur + dt

        return x_t

    # TODO: #19258: Pi0.5 calibration data should be available as a dataset

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: PretrainedCollectionModel,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        if component_name == "token_emb":
            raise NotImplementedError("token_emb is not quantized")

        if num_samples is None:
            num_samples = 100

        if component_name == "vision_encoder":
            return cls._calibration_data_vision_encoder(num_samples)
        if component_name == "backbone":
            return cls._calibration_data_backbone(num_samples)
        if component_name == "action_expert":
            return cls._calibration_data_action_expert(num_samples)
        raise ValueError(
            f"Unknown component_name={component_name!r}. "
            "Expected one of: vision_encoder, token_emb, backbone, action_expert."
        )

    @classmethod
    def _calibration_data_vision_encoder(cls, num_samples: int) -> DatasetEntries:
        cache_path = ASSET_CONFIG.get_local_store_dataset_path(
            "libero_vision_calib", "v1", f"data_n{num_samples}.pt"
        )
        if cache_path.exists():
            data = torch.load(cache_path, weights_only=True)
        else:
            dataset = LeRobotDataset("HuggingFaceVLA/libero")
            first_sample = dataset[0]
            image_keys = sorted(
                k for k in first_sample if k.startswith("observation.images.")
            )

            images: list[torch.Tensor] = []
            for idx in range(min(num_samples, len(dataset))):
                sample = dataset[idx] if idx > 0 else first_sample
                img = sample[image_keys[0]]
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                img = resize_and_normalize(img.to(torch.float32))
                images.append(img.squeeze(0))

            data = torch.stack(images[:num_samples])
            os.makedirs(cache_path.parent, exist_ok=True)
            torch.save(data, cache_path)

        return make_hub_dataset_entries(
            (_unbatch(data),),
            ["image"],
        )

    @classmethod
    def _calibration_data_backbone(cls, num_samples: int) -> DatasetEntries:
        cache_path = ASSET_CONFIG.get_local_store_dataset_path(
            "libero_backbone_calib", "v1", f"data_n{num_samples}.pt"
        )
        if cache_path.exists():
            data = torch.load(cache_path, weights_only=True)
        else:
            # Circular import: demo.py → pi05.__init__ → app.py
            from qai_hub_models.models.pi05.demo import (
                _build_preprocessed_batch,
                _to_device_tree,
            )

            dataset = LeRobotDataset("HuggingFaceVLA/libero")
            policy: PI05Policy = load_checkpoint(DEFAULT_CHECKPOINT)
            pi05_config: PI05Config = policy.config

            vit = Pi05PaliGemmaVision(policy).cpu().eval()
            token_emb = Pi05PaliGemmaTokenEmbed(policy).cpu().eval()

            image_keys = sorted(
                k
                for k, v in policy.model.config.input_features.items()
                if v.type == FeatureType.VISUAL and "empty" not in k
            )

            hidden_states = []
            att_masks = []
            rope_sins = []
            rope_coss = []

            n = min(num_samples, len(dataset))
            for idx in range(n):
                raw_sample = dataset[idx]
                raw_batch: dict = {}
                for k, v in raw_sample.items():
                    if isinstance(v, torch.Tensor):
                        raw_batch[k] = v.unsqueeze(0)
                    elif isinstance(v, str):
                        raw_batch[k] = [v]
                    else:
                        raw_batch[k] = v

                batch, _ = _build_preprocessed_batch(
                    cfg=pi05_config,
                    raw_batch=raw_batch,
                    batch_size=1,
                    dataset_stats=dataset.meta.stats,
                )
                batch = _to_device_tree(batch, "cpu")

                lang_tokens = batch["observation.language.tokens"]
                lang_mask = batch["observation.language.attention_mask"].to(
                    torch.float32
                )

                with torch.no_grad():
                    img_embeds = []
                    for key in image_keys:
                        img = batch[key]
                        if img.ndim != 4:
                            continue
                        img = resize_and_normalize(img)
                        img_embeds.append(vit(img))

                    (
                        prefix_emb,
                        prefix_att_2d,
                        prefix_sin,
                        prefix_cos,
                        _suffix_sin,
                        _suffix_cos,
                        _full_att_4d,
                    ) = token_emb(lang_tokens, lang_mask, *img_embeds)

                hidden_states.append(prefix_emb[0])
                att_masks.append(prefix_att_2d[0])
                rope_sins.append(prefix_sin[0])
                rope_coss.append(prefix_cos[0])

            data = {
                "hidden_state": torch.stack(hidden_states),
                "prefix_att_2d_masks": torch.stack(att_masks),
                "rope_emb_sin": torch.stack(rope_sins),
                "rope_emb_cos": torch.stack(rope_coss),
            }
            os.makedirs(cache_path.parent, exist_ok=True)
            torch.save(data, cache_path)

        return make_hub_dataset_entries(
            (
                _unbatch(data["hidden_state"]),
                _unbatch(data["prefix_att_2d_masks"]),
                _unbatch(data["rope_emb_sin"]),
                _unbatch(data["rope_emb_cos"]),
            ),
            ["hidden_state", "prefix_att_2d_mask", "rope_emb_sin", "rope_emb_cos"],
        )

    @classmethod
    def _calibration_data_action_expert(cls, num_samples: int) -> DatasetEntries:
        cache_path = ASSET_CONFIG.get_local_store_dataset_path(
            "libero_action_expert_calib", "v1", f"data_n{num_samples}.pt"
        )
        prefixes: list[dict[str, torch.Tensor]]
        steps: list[tuple[int, torch.Tensor, torch.Tensor]]
        if cache_path.exists():
            raw = torch.load(cache_path, weights_only=True)
            prefixes = raw["prefixes"]
            steps = raw["steps"]
        else:
            # Circular import: demo.py → pi05.__init__ → app.py
            from qai_hub_models.models.pi05.demo import (
                _build_preprocessed_batch,
                _to_device_tree,
            )

            dataset = LeRobotDataset("HuggingFaceVLA/libero")
            policy: PI05Policy = load_checkpoint(DEFAULT_CHECKPOINT)
            pi05_config: PI05Config = policy.config

            vit = Pi05PaliGemmaVision(policy).cpu().eval()
            token_emb = Pi05PaliGemmaTokenEmbed(policy).cpu().eval()
            backbone_full = Pi05PaliGemmaBackbone(policy).cpu().eval()
            action_expert = Pi05ActionExpert(policy).cpu().eval()

            image_keys = sorted(
                k
                for k, v in policy.model.config.input_features.items()
                if v.type == FeatureType.VISUAL and "empty" not in k
            )

            num_steps = int(getattr(pi05_config, "num_inference_steps", 10))
            state_dim = 32

            prefixes = []
            steps = []

            n = min(num_samples, len(dataset))
            for idx in range(n):
                raw_sample = dataset[idx]
                raw_batch: dict = {}
                for k, v in raw_sample.items():
                    if isinstance(v, torch.Tensor):
                        raw_batch[k] = v.unsqueeze(0)
                    elif isinstance(v, str):
                        raw_batch[k] = [v]
                    else:
                        raw_batch[k] = v

                batch, _ = _build_preprocessed_batch(
                    cfg=pi05_config,
                    raw_batch=raw_batch,
                    batch_size=1,
                    dataset_stats=dataset.meta.stats,
                )
                batch = _to_device_tree(batch, "cpu")

                lang_tokens = batch["observation.language.tokens"]
                lang_mask = batch["observation.language.attention_mask"].to(
                    torch.float32
                )

                with torch.no_grad():
                    img_embeds = []
                    for key in image_keys:
                        img = batch[key]
                        if img.ndim != 4:
                            continue
                        img = resize_and_normalize(img)
                        img_embeds.append(vit(img))

                    (
                        prefix_emb,
                        prefix_att_2d,
                        prefix_sin,
                        prefix_cos,
                        suffix_sin,
                        suffix_cos,
                        full_att_4d,
                    ) = token_emb(lang_tokens, lang_mask, *img_embeds)

                    rest_full = backbone_full(
                        prefix_emb, prefix_att_2d, prefix_sin, prefix_cos
                    )
                    n_full = len(rest_full) // 2
                    k_caches = list(rest_full[:n_full])
                    v_caches = list(rest_full[n_full:])

                prefix: dict[str, torch.Tensor] = {
                    "full_att_4d": full_att_4d[0],
                    "rope_emb_sin": suffix_sin[0],
                    "rope_emb_cos": suffix_cos[0],
                }
                for i in range(len(k_caches)):
                    prefix[f"key_cache_l{i}"] = k_caches[i][0]
                    prefix[f"value_cache_l{i}"] = v_caches[i][0]
                ep_idx = len(prefixes)
                prefixes.append(prefix)

                x_t = torch.randn(1, NUM_ACTION_STEPS, state_dim)
                dt = -1.0 / float(num_steps)
                t_cur = 1.0
                with torch.no_grad():
                    for _ in range(num_steps):
                        time_step = torch.tensor(t_cur)
                        steps.append((ep_idx, x_t[0].clone(), time_step))

                        kv_kwargs: dict[str, torch.Tensor] = {}
                        for i in range(len(k_caches)):
                            kv_kwargs[f"key_cache_l{i}"] = k_caches[i]
                            kv_kwargs[f"value_cache_l{i}"] = v_caches[i]

                        v_t = action_expert._compute_update(
                            full_att_4d=full_att_4d,
                            rope_emb_sin=suffix_sin,
                            rope_emb_cos=suffix_cos,
                            x_t=x_t,
                            time_step=torch.tensor([t_cur]),
                            **kv_kwargs,
                        )
                        x_t = x_t + dt * v_t
                        t_cur += dt

            os.makedirs(cache_path.parent, exist_ok=True)
            torch.save({"prefixes": prefixes, "steps": steps}, cache_path)

        input_names = [
            "full_att_4d",
            "rope_emb_sin",
            "rope_emb_cos",
            "x_t",
            "time_step",
        ]
        input_names.extend(f"key_cache_l{i}" for i in range(18))
        input_names.extend(f"value_cache_l{i}" for i in range(18))

        tensors_per_input: list[list[torch.Tensor | np.ndarray]] = [
            [] for _ in input_names
        ]
        for ep_idx, x_t, time_step in steps:
            prefix = prefixes[ep_idx]
            tensors_per_input[0].append(prefix["full_att_4d"].unsqueeze(0))
            tensors_per_input[1].append(prefix["rope_emb_sin"].unsqueeze(0))
            tensors_per_input[2].append(prefix["rope_emb_cos"].unsqueeze(0))
            tensors_per_input[3].append(x_t.unsqueeze(0))
            tensors_per_input[4].append(time_step.unsqueeze(0))
            for i in range(18):
                tensors_per_input[5 + i].append(prefix[f"key_cache_l{i}"].unsqueeze(0))
            for i in range(18):
                tensors_per_input[23 + i].append(
                    prefix[f"value_cache_l{i}"].unsqueeze(0)
                )

        return make_hub_dataset_entries(tuple(tensors_per_input), input_names)
