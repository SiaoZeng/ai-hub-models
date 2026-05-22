# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

import pandas as pd
import torch

from qai_hub_models.models._shared.video_classifier.utils import (
    DEFAULT_NUM_CLIPS,
    DEFAULT_NUM_CROPS,
    get_class_name_kinetics_400,
    multi_crop,
    preprocess_video_224,
    preprocess_video_kinetics_400,
    read_video_at_fps,
    read_video_per_second,
    sample_clips,
    sample_video,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec

KINETICS400_FOLDER_NAME = "kinetics400"
KINETICS400_VERSION = 2

# Some of the video files in the training data downloaded from the Internet
# have corrupted video files that can't be opened. If we try to load these samples
# at runtime, it throws an error and kills the process. We decided it best to remove
# these files right after downloading to avoid this error.
CORRUPTED_TRAIN_VIDEOS = [
    "-I4Ggi6-QOE_000054_000064.mp4",
    "-K_uevUt2V8_000003_000013.mp4",
    "-6wkYqjFei0_000015_000025.mp4",
    "-IT7W5_Y3Gc_000003_000013.mp4",
    "-4c4r9YeS6s_000098_000108.mp4",
    "--PyMoD3_eg_000020_000030.mp4",
    "-3pp5xan1Hw_000006_000016.mp4",
]


def _get_labeled_data(
    videos_folder: Path, labels_csv_path: Path
) -> tuple[list[str], list[int]]:
    """
    Given the folder with a subset of videos and the appropriate labels file,
    returns a list of filenames and a list of label indices for the subset that match.
    """
    video_metadata_rows = []
    for filename in os.listdir(videos_folder):
        filename_split = filename[: -len(".mp4")].split("_")
        youtube_id = "_".join(filename_split[:-2])
        start = int(filename_split[-2])
        end = int(filename_split[-1])
        video_metadata_rows.append((youtube_id, start, end))
    video_metadata_df = pd.DataFrame(
        video_metadata_rows, columns=["youtube_id", "time_start", "time_end"]
    )
    labels_df = pd.read_csv(labels_csv_path)
    join_df = labels_df.merge(
        video_metadata_df, on=["youtube_id", "time_start", "time_end"], how="inner"
    )

    # Sort to ensure deterministic ordering regardless of filesystem
    join_df = join_df.sort_values(by=["youtube_id", "time_start"])
    video_paths: list[str] = []
    label_indices: list[int] = []
    label_index_map = {
        label: i for (i, label) in enumerate(get_class_name_kinetics_400())
    }
    for _, row in join_df.iterrows():
        assert isinstance(row, pd.Series)
        video_paths.append(
            f"{row.youtube_id}_{row.time_start:06d}_{row.time_end:06d}.mp4"
        )
        label_indices.append(label_index_map[row.label])
    return video_paths, label_indices


class PreprocessProtocol(Enum):
    """
    Per-clip eval preprocessing recipe.

    Selects the family of decode + resize + crop steps applied to each clip
    before it is fed to the model. Picked explicitly by the caller (or
    inferred once at ``__init__`` from the model's input spatial size) so
    later code paths don't have to re-derive intent from a raw shape.

    KINETICS_112_TORCHVISION
        torchvision R3D / R2+1D / MC3 protocol: read at 15 fps, resize to
        (128, 171), center-crop to 112x112.
    KINETICS_224_VIDEOMAE
        VideoMAE protocol: read at native fps, short-side resize to 256;
        center-crop to 224x224 only when ``multi_crop`` won't run after.
    """

    KINETICS_112_TORCHVISION = "kinetics_112_torchvision"
    KINETICS_224_VIDEOMAE = "kinetics_224_videomae"


class Kinetics400Dataset(BaseDataset):
    """
    Class for using the Kinetics400 dataset for video classification:
        https://github.com/cvdfoundation/kinetics-dataset
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        num_frames: int = 16,
        input_spec: InputSpec | None = None,
        num_clips: int = DEFAULT_NUM_CLIPS,
        num_crops: int = DEFAULT_NUM_CROPS,
    ) -> None:
        self.num_frames = num_frames
        self.num_clips = num_clips
        self.num_crops = num_crops
        self.split_str = split.name.lower()
        self.videos_asset = CachedWebDatasetAsset(
            f"https://s3.amazonaws.com/kinetics/400/{self.split_str}/part_0.tar.gz",
            KINETICS400_FOLDER_NAME,
            KINETICS400_VERSION,
            os.path.join(self.split_str, "part_0.tar.gz"),
        )
        self.csv_asset = CachedWebDatasetAsset(
            f"https://s3.amazonaws.com/kinetics/400/annotations/{self.split_str}.csv",
            KINETICS400_FOLDER_NAME,
            KINETICS400_VERSION,
            os.path.join("annotations", f"{self.split_str}.csv"),
        )
        self.videos_folder = self.videos_asset.extracted_path
        self.video_dim = input_spec["video"][0][-1] if input_spec else 112
        assert self.video_dim in [112, 224], "Video dimension must be 112 or 224."
        if self.video_dim == 112:
            self.protocol = PreprocessProtocol.KINETICS_112_TORCHVISION
        else:
            self.protocol = PreprocessProtocol.KINETICS_224_VIDEOMAE
        BaseDataset.__init__(
            self, str(self.videos_folder), split=split, input_spec=input_spec
        )

    def __len__(self) -> int:
        # Number of non-corrupted videos in each split
        return 993 if self.split == DatasetSplit.TRAIN else 1000

    def _validate_data(self) -> bool:
        if not self.csv_asset.path.exists():
            return False

        videos_folder = self.videos_asset.path.parent
        if not videos_folder.exists():
            return False

        self.mp4_files, self.label_indices = _get_labeled_data(
            self.videos_folder, self.csv_asset.path
        )

        if len(self.mp4_files) != len(self):
            return False

        return len(self.label_indices) == len(self)

    def preprocess_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the per-clip preprocessing for ``self.protocol``."""
        if self.protocol is PreprocessProtocol.KINETICS_112_TORCHVISION:
            return preprocess_video_kinetics_400(tensor)
        # KINETICS_224_VIDEOMAE: skip the center-crop iff multi_crop will
        # take over the spatial-view step afterwards.
        return preprocess_video_224(tensor, center_crop=self.num_crops == 1)

    def __getitem__(
        self, video_idx: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Decode the video at ``video_idx``, build all ``num_clips * num_crops``
        views, and return them packed for the model.

        Views are packed along the channel dim as ``[num_views*3, T, H, W]``;
        the model's ``forward`` splits them back with an explicit reshape.
        ``video_id`` is returned alongside the label so the evaluator can
        aggregate scores across views belonging to the same video.

        Parameters
        ----------
        video_idx
            Index of the video in ``[0, num_videos)``.

        Returns
        -------
        views : torch.Tensor
            Shape ``[num_views*3, T, H, W]``.
        gt : tuple[torch.Tensor, torch.Tensor]
            ``(label, video_id)`` as scalar ``int64`` tensors.
        """
        video_path = str(self.videos_folder / self.mp4_files[video_idx])
        if self.protocol is PreprocessProtocol.KINETICS_112_TORCHVISION:
            raw_video = read_video_at_fps(video_path, target_fps=15)
        else:
            raw_video = read_video_per_second(video_path)

        all_clips: list[torch.Tensor | None]
        if self.num_clips > 1:
            all_clips = list(sample_clips(raw_video, self.num_frames, self.num_clips))
        else:
            all_clips = [sample_video(raw_video, self.num_frames)]
        del raw_video

        # Fill merged output per-clip so each preprocessed clip + its crops
        # can be freed before the next iteration, avoiding peak memory of
        # all views simultaneously.
        merged = torch.empty(
            self.num_clips * self.num_crops * 3,
            self.num_frames,
            self.video_dim,
            self.video_dim,
            dtype=torch.float32,
        )

        view_idx = 0
        for idx, clip in enumerate(all_clips):
            assert clip is not None
            preprocessed = self.preprocess_tensor(clip)
            all_clips[idx] = None
            if self.num_crops > 1:
                crops = multi_crop(preprocessed, self.video_dim, self.num_crops)
            else:
                crops = [preprocessed]
            del preprocessed
            for crop in crops:
                merged[view_idx * 3 : (view_idx + 1) * 3].copy_(crop)
                view_idx += 1
            del crops

        label = torch.tensor(self.label_indices[video_idx], dtype=torch.int64)
        video_id = torch.tensor(video_idx, dtype=torch.int64)
        return merged, (label, video_id)

    def _download_data(self) -> None:
        self.videos_asset.fetch(extract=True)
        self.csv_asset.fetch()

        if self.split == DatasetSplit.TRAIN:
            for video in CORRUPTED_TRAIN_VIDEOS:
                os.remove(self.videos_folder / video)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 40

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://github.com/cvdfoundation/kinetics-dataset",
            split_description="part 0 of the validation split",
        )
