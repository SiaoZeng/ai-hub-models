# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json
from copy import deepcopy

import onnx

from qai_hub_models.utils.onnx.helpers import ONNXBundle


def propagate_memory_encodings(
    encodings: dict[str, dict],
    model: onnx.ModelProto,
) -> None:
    """
    Propagate encodings through memory ops. This can be important if the
    model will be split up into multiple parts if the split points run
    through ops that do not have encodings and rely on propagation
    downstream. Encodings are updated in place.
    """
    changes = True
    while changes:
        changes = False
        for node in model.graph.node:
            if node.output[0] in encodings["activation_encodings"]:
                continue

            if (
                node.op_type
                in {
                    "Concat",
                    "Split",
                    "Transpose",
                    "Cast",
                    "Reshape",
                    "Slice",
                    "Squeeze",
                    "Unsqueeze",
                    "Expand",
                }
                and node.input[0] in encodings["activation_encodings"]
            ):
                for output_name in node.output:
                    dst_entry = deepcopy(
                        encodings["activation_encodings"][node.input[0]]
                    )
                    if isinstance(dst_entry, dict):
                        dst_entry["name"] = output_name
                    encodings["activation_encodings"][output_name] = dst_entry
                    changes = True


def apply_propagate_memory_encodings(bundle: ONNXBundle) -> None:
    """Load encodings + ONNX from *bundle*, propagate memory encodings,
    and write back.
    """
    encodings_path = bundle.aimet_encodings_path
    assert encodings_path is not None, f"No encodings found in {bundle.bundle_path}"

    with open(encodings_path) as f:
        encodings = json.load(f)

    assert isinstance(encodings.get("activation_encodings"), list)
    encodings["activation_encodings"] = {
        v["name"]: v for v in encodings["activation_encodings"]
    }
    encodings["param_encodings"] = {v["name"]: v for v in encodings["param_encodings"]}

    model = onnx.load(str(bundle.onnx_graph_path))
    propagate_memory_encodings(encodings, model)

    encodings["activation_encodings"] = list(encodings["activation_encodings"].values())
    encodings["param_encodings"] = list(encodings["param_encodings"].values())

    with open(encodings_path, "w") as f:
        json.dump(encodings, f, indent=4, sort_keys=True)
