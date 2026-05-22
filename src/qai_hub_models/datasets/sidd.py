# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from os import path as osp

import cv2
import lmdb
import numpy as np
import torch

from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.image_processing import numpy_image_to_torch
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

SIDD_FOLDER_NAME = "sidd"
SIDD_VERSION = 1

SIDD_PRIVATE_ASSET = CachedPrivateDatasetAsset(
    f"qai-hub-models/datasets/{SIDD_FOLDER_NAME}/v{SIDD_VERSION}/SIDD-val-lmdb.zip",
    SIDD_FOLDER_NAME,
    SIDD_VERSION,
    "SIDD-val-lmdb.zip",
    installation_steps=[
        "Download the SIDD-val-lmdb.zip file from https://drive.google.com/file/d/1gZx_K2vmiHalRNOb1aj93KuUQ2guOlLp/view",
        "Run `python -m qai_hub_models.datasets.configure_dataset --dataset sidd --files /path/to/SIDD-val-lmdb.zip`",
    ],
)


class SIDDDataset(BaseDataset):
    """SIDD (Smartphone Image Denoising Dataset)"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_data_zip: str | None = None,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.images_path = SIDD_PRIVATE_ASSET.extracted_path
        self.gt_folder = str(self.images_path / "SIDD" / "val" / "gt_crops.lmdb")
        self.lq_folder = str(self.images_path / "SIDD" / "val" / "input_crops.lmdb")
        self.input_data_zip = input_data_zip
        BaseDataset.__init__(self, self.images_path, split, input_spec)
        # Initialize LMDB environments and transactions directly

        self.lq_env = lmdb.open(
            self.lq_folder, readonly=True, lock=False, readahead=False, meminit=False
        )
        self.gt_env = lmdb.open(
            self.gt_folder, readonly=True, lock=False, readahead=False, meminit=False
        )

        # Create persistent read transactions
        self.lq_txn = self.lq_env.begin(write=False)
        self.gt_txn = self.gt_env.begin(write=False)

        # Get paired paths from LMDB
        self.paths = _paired_paths_from_lmdb(
            [self.lq_folder, self.gt_folder], ["lq", "gt"]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        gt_path = self.paths[index]["gt_path"]
        img_gt = _get_image_from_lmdb(self.gt_txn, gt_path, "gt", float32=True)

        lq_path = self.paths[index]["lq_path"]
        img_lq = _get_image_from_lmdb(self.lq_txn, lq_path, "lq", float32=True)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt_tensor = numpy_image_to_torch(
            cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB), to_float=False
        )
        img_lq_tensor = numpy_image_to_torch(
            cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB), to_float=False
        )

        return img_lq_tensor.squeeze(0), img_gt_tensor.squeeze(0)

    def __del__(self) -> None:
        """Clean up LMDB resources"""
        if hasattr(self, "lq_txn"):
            self.lq_txn.abort()
        if hasattr(self, "gt_txn"):
            self.gt_txn.abort()
        if hasattr(self, "lq_env"):
            self.lq_env.close()
        if hasattr(self, "gt_env"):
            self.gt_env.close()

    def _validate_data(self) -> bool:
        if not self.images_path.exists():
            return False
        # Fast path: expected structure
        if osp.exists(self.gt_folder) and osp.exists(self.lq_folder):
            return True
        # Fallback: search recursively for the gt_crops.lmdb directory.
        # Handles zips with different top-level directory structures.
        for gt_dir in sorted(self.images_path.rglob("gt_crops.lmdb")):
            lq_candidate = gt_dir.parent / "input_crops.lmdb"
            if gt_dir.is_dir() and lq_candidate.exists():
                self.gt_folder = str(gt_dir)
                self.lq_folder = str(lq_candidate)
                return True
        return False

    def _download_data(self) -> None:
        SIDD_PRIVATE_ASSET.fetch(extract=True, local_path=self.input_data_zip)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 50

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www.eecs.yorku.ca/~kamel/sidd/",
            split_description="SIDD dataset for image denoising",
        )


def _paired_paths_from_lmdb(
    folders: list[str], keys: list[str]
) -> list[dict[str, str]]:
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert len(keys) == 2, (
        f"The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}"
    )
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith(".lmdb") and gt_folder.endswith(".lmdb")):
        raise ValueError(
            f"{input_key} folder and {gt_key} folder should both in lmdb "
            f"formats. But received {input_key}: {input_folder}; "
            f"{gt_key}: {gt_folder}"
        )
    # ensure that the two meta_info files are the same
    with open(osp.join(input_folder, "meta_info.txt")) as fin:
        input_lmdb_keys = [line.split(".")[0] for line in fin]
    with open(osp.join(gt_folder, "meta_info.txt")) as fin:
        gt_lmdb_keys = [line.split(".")[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(
            f"Keys in {input_key}_folder and {gt_key}_folder are different."
        )
    return [
        {f"{input_key}_path": lmdb_key, f"{gt_key}_path": lmdb_key}
        for lmdb_key in sorted(input_lmdb_keys)
    ]


def _imfrombytes(content: bytes, float32: bool = False) -> np.ndarray:
    img_np = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    if float32:
        img = img.astype(np.float32) / 255.0
    return img


def _get_image_from_lmdb(
    txn: lmdb.Transaction, path: str, key_name: str, float32: bool = True
) -> np.ndarray:
    img_bytes = txn.get(path.encode("ascii"))
    if img_bytes is None:
        raise ValueError(f"Key not found: {path}")
    try:
        return _imfrombytes(img_bytes, float32=float32)
    except ValueError as exc:
        raise RuntimeError(f"{key_name} path {path} not working") from exc
