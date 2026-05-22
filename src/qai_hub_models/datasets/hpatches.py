# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.image_processing import preprocess_PIL_image
from qai_hub_models.utils.input_spec import InputSpec

HPATCHES_URL = "https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip"
HPATCHES_FOLDER_NAME = "hpatches"
HPATCHES_VERSION = 2
HPATCHES_ASSET = CachedWebDatasetAsset(
    HPATCHES_URL,
    HPATCHES_FOLDER_NAME,
    HPATCHES_VERSION,
    "hpatches-sequences-release.zip",
)

# Number of images per sequence (1 reference + 5 target)
IMAGES_PER_SEQ = 6
# Total sequences in the full dataset
NUM_SEQUENCES = 116


class HPatchesDataset(BaseDataset):
    """
    HPatches dataset for homography estimation evaluation.
    https://github.com/hpatches/hpatches-dataset

    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        """
        Parameters
        ----------
        split
            Accepted for interface compatibility with BaseDataset but ignored:
            HPatches has no train/val/test split — the full dataset is always loaded.
        input_spec
            Optional input specification. If provided, height and width are
            extracted from the ``image`` entry (shape ``(B, N, 1, H, W)``).
            Defaults to 480x640 if not provided.
        """
        input_spec = input_spec or {"image": ((1, 2, 1, 480, 640), "float32")}
        self.height = input_spec["image"][0][3]
        self.width = input_spec["image"][0][4]
        self.hpatches_path = HPATCHES_ASSET.extracted_path
        super().__init__(self.hpatches_path, split, input_spec)
        self.hpatches_path = self._find_sequences_root(self.hpatches_path)
        self._build_pairs()

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Custom collate function for HPatches dataset.

        Must be passed explicitly when constructing a DataLoader::

            DataLoader(dataset, collate_fn=HPatchesDataset.collate_fn)

        Parameters
        ----------
        batch
            List of tuples ``(images, H_gt)`` as returned by ``__getitem__``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            images ``(B, 2, 1, H, W)`` and H_gt ``(B, 3, 3)``.
        """
        images_list = [item[0] for item in batch]
        H_gt_list = [item[1] for item in batch]

        images = torch.stack(images_list, dim=0)
        H_gt = torch.stack(H_gt_list, dim=0)

        return images, H_gt

    def _download_data(self) -> None:
        HPATCHES_ASSET.fetch(extract=True)

    @staticmethod
    def _is_sequence_dir(d: Path) -> bool:
        """Check if directory is an HPatches sequence (i_* or v_* prefix)."""
        return d.is_dir() and (d.name.startswith("i_") or d.name.startswith("v_"))

    def _validate_data(self) -> bool:
        if not HPATCHES_ASSET.extracted_path.exists():
            return False
        root = self._find_sequences_root(HPATCHES_ASSET.extracted_path)
        seqs = [d for d in root.iterdir() if self._is_sequence_dir(d)]
        return len(seqs) == NUM_SEQUENCES

    def _find_sequences_root(self, path: Path) -> Path:
        """
        Walk down from ``path`` until we find a directory that directly
        contains HPatches sequence folders (``i_*`` / ``v_*``).
        """
        current = path
        for _ in range(3):  # at most 3 levels deep
            entries = list(current.iterdir())
            seq_dirs = [e for e in entries if self._is_sequence_dir(e)]
            if seq_dirs:
                return current
            # Descend into the single sub-directory if there is exactly one
            sub_dirs = [e for e in entries if e.is_dir()]
            if len(sub_dirs) == 1:
                current = sub_dirs[0]
            else:
                break
        return current

    @classmethod
    def dataset_name(cls) -> str:
        return "hpatches"

    @staticmethod
    def default_samples_per_job() -> int:
        return 580

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://github.com/hpatches/hpatches-dataset",
            split_description="full dataset (116 sequences x 5 pairs)",
        )

    def _build_pairs(self) -> None:
        """
        Build the list of (seq_path, target_idx) pairs.

        Each sequence contributes 5 pairs: (image1, image2..6).
        """
        seqs = sorted(
            d for d in self.hpatches_path.iterdir() if self._is_sequence_dir(d)
        )
        self._pairs: list[tuple[Path, int]] = []
        for seq in seqs:
            for idx in range(2, IMAGES_PER_SEQ + 1):
                self._pairs.append((seq, idx))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a stacked pair of images and the ground-truth homography.

        Parameters
        ----------
        item
            Dataset index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            images ``(2, 1, H, W)`` float32 in ``[0, 1]`` (images[0] is
            reference, images[1] is warped), H_gt ``(3, 3)`` homography
            rescaled to evaluation resolution.
        """
        seq_path, tgt_idx = self._pairs[item]

        with (
            Image.open(seq_path / "1.ppm") as pil0,
            Image.open(seq_path / f"{tgt_idx}.ppm") as pil1,
        ):
            orig_w0, orig_h0 = pil0.size
            orig_w1, orig_h1 = pil1.size

            image0 = preprocess_PIL_image(
                pil0.convert("L").resize(
                    (self.width, self.height), resample=Image.BILINEAR
                )
            )[0]
            image1 = preprocess_PIL_image(
                pil1.convert("L").resize(
                    (self.width, self.height), resample=Image.BILINEAR
                )
            )[0]

        H_raw = np.loadtxt(str(seq_path / f"H_1_{tgt_idx}"), dtype=np.float64)
        H_gt = self._rescale_homography(
            H_raw,
            src_hw=(orig_h0, orig_w0),
            dst_hw=(orig_h1, orig_w1),
        )

        images = torch.stack([image0, image1], dim=0)
        return images, H_gt

    def _rescale_homography(
        self,
        H: np.ndarray,
        src_hw: tuple[int, int],
        dst_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Rescale a raw homography from original resolution to evaluation resolution."""
        src_h, src_w = src_hw
        dst_h, dst_w = dst_hw
        eval_h, eval_w = self.height, self.width

        S_src = np.array(
            [[eval_w / src_w, 0, 0], [0, eval_h / src_h, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        S_dst = np.array(
            [[eval_w / dst_w, 0, 0], [0, eval_h / dst_h, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        H_resized = S_dst @ H @ np.linalg.inv(S_src)
        return torch.from_numpy(H_resized.astype(np.float32))
