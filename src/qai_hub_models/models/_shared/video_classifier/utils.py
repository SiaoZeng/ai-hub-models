# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import torch
from torchvision.transforms import functional as TF

from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT


def normalize(video: torch.Tensor) -> torch.Tensor:
    """
    Normalize the video frames.

    Parameters
    ----------
    video
        Video tensor (Number of frames x HWC) with values between 0-255.
        Channel Layout: RGB

    Returns
    -------
    normalized_video : torch.Tensor
        Video is normalized to have values between 0-1
        and transposed so the shape is Channel x Number of frames x HW.
    """
    return video.permute(3, 0, 1, 2).to(torch.float32) / 255


def sample_video(video: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    Samples the number of frames in the video to the number requested.

    Parameters
    ----------
    video
        A [T, H, W, C] video tensor.
    num_frames
        Number of frames to sample video down to.

    Returns
    -------
    sampled_video : torch.Tensor
        Video tensor sampled to the appropriate number of frames.
    """
    total = video.shape[0]
    if total == 0:
        raise ValueError("Video has no frames.")
    if total <= num_frames:
        pad = video[-1:].expand(num_frames - total, *video.shape[1:])
        return torch.cat([video, pad], dim=0)
    frame_rate = max(1, total // num_frames)
    indices = torch.arange(num_frames) * frame_rate
    indices = indices.clamp(max=total - 1)
    return video[indices]


def sample_clips(
    video: torch.Tensor, num_frames: int, num_clips: int
) -> list[torch.Tensor]:
    """
    Uniformly sample ``num_clips`` temporal clips from a video, each
    containing ``num_frames`` **consecutive** frames.

    Applies the same linspace-and-floor formula as torchvision's
    ``UniformClipSampler`` (in ``torchvision.datasets.samplers``) but on a
    pre-decoded ``[T, H, W, C]`` tensor. We don't call the stock sampler
    directly because it's a ``torch.utils.data.Sampler`` over a
    ``VideoClips`` object whose ``get_clip()`` decodes via
    ``torchvision.io.read_video`` — that import is missing in the qaihm
    build env (we hit it in CI) and torchvision deprecated the video IO
    backend in 0.22 (removed in 0.24), while our ``requirements.txt``
    allows torchvision up to 0.26. Reimplementing the formula keeps the
    decode path pluggable (torchcodec fallback) while matching the
    reference R3D/R2+1D/MC3 eval protocol.

    Formula: ``torch.linspace(0, num_windows - 1, steps=num_clips).floor()``
    where ``num_windows = max(1, total_frames - num_frames + 1)``.

    Parameters
    ----------
    video
        Raw video tensor of shape ``[T, H, W, C]``.
    num_frames
        Number of consecutive frames per clip.
    num_clips
        Number of temporal clips to extract.

    Returns
    -------
    clips : list[torch.Tensor]
        List of ``num_clips`` tensors, each of shape
        ``[num_frames, H, W, C]``.
    """
    total_frames = video.shape[0]
    num_windows = max(1, total_frames - num_frames + 1)
    # Exactly matches UniformClipSampler: linspace over window indices, floor
    starts = torch.linspace(0, num_windows - 1, steps=num_clips).floor().long()
    clips: list[torch.Tensor] = []
    for start in starts.tolist():
        start = int(start)
        clip = video[start : start + num_frames]
        if clip.shape[0] < num_frames:
            pad = clip[-1:].expand(num_frames - clip.shape[0], *clip.shape[1:])
            clip = torch.cat([clip, pad], dim=0)
        clips.append(clip)
    return clips


def multi_crop(
    video: torch.Tensor, crop_size: int, num_crops: int
) -> list[torch.Tensor]:
    """
    Extract ``num_crops`` spatial crops from a video clip along the longer
    spatial axis.  This mirrors the multi-crop evaluation protocol used in
    published benchmarks (e.g. 3 crops: left/center/right or top/center/bottom).

    The video is assumed to have already been resized so that its shorter side
    equals ``crop_size``; crops are taken along the longer side.

    Parameters
    ----------
    video
        Video tensor of shape ``[C, T, H, W]`` (channel-first, after
        ``normalize`` has been applied).
    crop_size
        Spatial size of each square crop (height == width == crop_size).
    num_crops
        Number of spatial crops to extract (typically 1 or 3).

    Returns
    -------
    crops : list[torch.Tensor]
        List of ``num_crops`` tensors, each of shape ``[C, T, crop_size, crop_size]``.
    """
    h, w = video.shape[-2], video.shape[-1]
    if h < crop_size or w < crop_size:
        raise ValueError(
            f"Video spatial dims ({h}x{w}) are smaller than crop_size "
            f"({crop_size}). Ensure the video is resized before calling "
            f"multi_crop()."
        )
    crops: list[torch.Tensor] = []
    for crop_idx in range(num_crops):
        if h > w:
            if num_crops == 1:
                i = (h - crop_size) // 2
            else:
                i = int(crop_idx * (h - crop_size) / max(num_crops - 1, 1))
            j = (w - crop_size) // 2
        else:
            i = (h - crop_size) // 2
            if num_crops == 1:
                j = (w - crop_size) // 2
            else:
                j = int(crop_idx * (w - crop_size) / max(num_crops - 1, 1))
        crops.append(video[..., i : i + crop_size, j : j + crop_size])
    return crops


def resize(video: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """
    Interpolate the frames of the image to match model's input resolution.

    Parameters
    ----------
    video
        Input video tensor.
    size
        Target size (height, width) for resizing.

    Returns
    -------
    resized_video : torch.Tensor
        Resized video is returned.
        Selected settings for resize were recommended.
    """
    return torch.nn.functional.interpolate(
        video, size=size, scale_factor=None, mode="bilinear", align_corners=False
    )


def crop(video: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    """
    Center crop the video frames.

    Parameters
    ----------
    video
        Input video torch.Tensor.
    output_size
        Desired output shape for each frame.

    Returns
    -------
    cropped_video : torch.Tensor
        Center cropped based on the output size.
    """
    h, w = video.shape[-2:]
    th, tw = output_size
    i = round((h - th) / 2.0)
    j = round((w - tw) / 2.0)
    return video[..., i : (i + th), j : (j + tw)]


def read_video_at_fps(path: str, target_fps: int = 15) -> torch.Tensor:
    """
    Read video from path and resample to ``target_fps``.

    The torchvision reference evaluation for R3D/R2+1D/MC3 uses
    ``frame_rate=15`` when building the dataset, so we resample every
    video to 15 fps before extracting clips.

    Parameters
    ----------
    path
        Path of the input video.
    target_fps
        Target frame rate to resample to (default: 15).

    Returns
    -------
    video_tensor : torch.Tensor
        Resampled video tensor of shape ``[T, H, W, C]``.
    """
    try:
        from torchvision.io import read_video as tv_read_video

        frames, _, info = tv_read_video(path, pts_unit="sec")
        native_fps = float(info.get("video_fps", 0.0))
    except ImportError:
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(path, dimension_order="NHWC")
        frames = decoder.get_all_frames().data
        native_fps = float(decoder.metadata.average_fps or 0.0)
    if native_fps <= 0 or abs(native_fps - target_fps) < 0.5:
        return frames
    step = native_fps / target_fps
    total = frames.shape[0]
    num_out = max(1, int(total / step))
    indices = (torch.arange(num_out) * step).round().long().clamp_(max=total - 1)
    return frames[indices]


def read_video_per_second(path: str) -> torch.Tensor:
    """
    Read video from path and convert to torch tensor at native fps.

    Parameters
    ----------
    path
        Path of the input video.

    Returns
    -------
    video_tensor : torch.Tensor
        Video tensor of shape ``[T, H, W, C]`` at native fps.
    """
    try:
        from torchvision.io import read_video as tv_read_video

        return tv_read_video(path)[0]
    except ImportError:
        from torchcodec.decoders import VideoDecoder

        return VideoDecoder(path, dimension_order="NHWC").get_all_frames().data


def preprocess_video_kinetics_400(input_video: torch.Tensor) -> torch.Tensor:
    """
    Preprocess the input video correctly for video classification inference.

    This is specific to torchvision models that take input of size 112.

    Sourced from: https://github.com/pytorch/vision/tree/main/references/video_classification

    Parameters
    ----------
    input_video
        Raw input tensor of shape [T, H, W, C], uint8 values 0-255.

    Returns
    -------
    preprocessed_video : torch.Tensor
        Shape [C, T, 112, 112], float32 in [0, 1], ready for the model
        which applies mean/std normalization in its forward pass.
    """
    input_video = normalize(input_video)
    input_video = resize(input_video, (128, 171))
    return crop(input_video, (112, 112))


def preprocess_video_224(
    input_video: torch.Tensor,
    short_side_size: int = 256,
    center_crop: bool = True,
) -> torch.Tensor:
    """
    Preprocess the input video correctly for video classification inference.

    This is specific to models like video_mae which take inputs of size 224.
    Resizes the shorter spatial side to ``short_side_size`` and, when
    ``center_crop`` is True, center-crops to 224x224.  Pass
    ``center_crop=False`` when the caller runs ``multi_crop`` afterwards.

    Sourced from: https://github.com/MCG-NJU/VideoMAE/blob/14ef8d856287c94ef1f985fe30f958eb4ec2c55d/kinetics.py#L56

    Parameters
    ----------
    input_video
        Raw input tensor of shape ``[T, H, W, C]``.
    short_side_size
        Size to which the shorter spatial side is resized.
    center_crop
        If True, center-crop to 224x224 after resize.

    Returns
    -------
    preprocessed_video : torch.Tensor
        Tensor of shape ``[C, T, 224, 224]`` when ``center_crop`` is True,
        otherwise ``[C, T, short_side_size, longer_side]``.
    """
    input_video = normalize(input_video)
    input_video = TF.resize(input_video, short_side_size)
    if center_crop:
        input_video = TF.center_crop(input_video, [224, 224])
    return input_video


def get_class_name_kinetics_400() -> list[str]:
    """
    Return the list of class names in the correct order, where the class index
    within this list corresponds to logit at the same index of the model output.

    Returns
    -------
    class_names : list[str]
        List of class names for Kinetics-400 dataset.
    """
    labels_path = QAIHM_PACKAGE_ROOT / "labels" / "kinetics400_labels.txt"
    with open(labels_path) as f:
        return [line.strip() for line in f]
