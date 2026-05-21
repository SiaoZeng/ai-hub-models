# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from qai_hub.client import Device

from qai_hub_models.configs.model_metadata import OutputSpec
from qai_hub_models.models.common import (
    Precision,
    SampleInputsType,
    TargetRuntime,
)
from qai_hub_models.models.protocols import FromPretrainedProtocol
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
)
from qai_hub_models.utils.export_result import (
    ComponentGroup,
    MultiGraphComponentGroup,
    MultiGraphGroup,
)
from qai_hub_models.utils.input_spec import InputSpec, make_torch_inputs
from qai_hub_models.utils.qai_hub_helpers import (
    build_compile_options,
    build_link_options,
    build_profile_options,
    build_quantize_options,
)
from qai_hub_models.utils.transpose_channel import transpose_channel_first_to_last

__all__ = [
    "MultiGraphCollectionModel",
    "MultiGraphWorkbenchModel",
]


class MultiGraphWorkbenchModel(FromPretrainedProtocol):
    """
    A model composed of multiple independently-compiled graphs that are linked into one executable binary.

    The QAIRT SDK can support a single model file that contains multiple graphs.
    The graphs will share weights on disk when possible.

    Subclasses must implement:
    - `get_graph_names()` — the source of truth for which graphs exist
    - `get_graph_input_spec(graph_name)` — input spec for a single graph
    - `get_graph_output_spec(graph_name)` — output spec for a single graph
    - `convert_to_hub_source_model(...)` — serialize to disk

    The `get_*` methods auto-build dicts from the per-graph `get_graph_*` getters.
    """

    # -- Subclasses must implement these --

    @property
    def graph_names(self) -> list[str]:
        raise NotImplementedError

    def get_graph_output_spec(self, graph_name: str) -> OutputSpec:
        raise NotImplementedError

    def get_graph_input_spec(
        self, graph_name: str, *args: Any, **kwargs: Any
    ) -> InputSpec:
        raise NotImplementedError

    @property
    def shared_source_model(self) -> bool:
        return False

    def convert_graph_to_hub_source_model(
        self,
        graph_name: str,
        output_path: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path | None:
        raise NotImplementedError

    # -- Subclasses may override these --

    def get_graph_channel_last_input(self, graph_name: str) -> list[str]:
        return []

    def get_graph_channel_last_output(self, graph_name: str) -> list[str]:
        return []

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> str | None:
        if target_runtime.is_aot_compiled:
            return "Multi-graph models only support AOT-compiled target runtimes."
        return None

    def component_precision(self) -> Precision:
        raise NotImplementedError()

    def get_graph_sample_inputs(
        self,
        graph_name: str,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
    ) -> SampleInputsType:
        spec = input_spec or self.get_graph_input_spec(graph_name)
        inputs_dict: SampleInputsType = {}
        inputs_list = make_torch_inputs(spec)
        for i, input_name in enumerate(spec.keys()):
            inputs_dict[input_name] = [inputs_list[i].numpy()]
        if use_channel_last_format and (
            cl := self.get_graph_channel_last_input(graph_name)
        ):
            inputs_dict = transpose_channel_first_to_last(cl, inputs_dict)
        return inputs_dict

    # -- Per-graph hub option getters (subclasses may override) --

    def get_graph_hub_litemp_percentage(
        self, graph_name: str, precision: Precision
    ) -> float | None:
        return None

    def get_graph_hub_quantize_options(
        self,
        graph_name: str,
        precision: Precision,
        other_quantize_options: str = "",
    ) -> str:
        litemp_percentage = (
            self.get_graph_hub_litemp_percentage(graph_name, precision)
            if precision.override_type is not None
            else None
        )
        return build_quantize_options(
            precision, litemp_percentage, other_quantize_options
        )

    def get_graph_hub_compile_options(
        self,
        graph_name: str,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        assert target_runtime.is_aot_compiled, (
            "Multi-graph models only support AOT-compiled target runtimes."
        )
        return build_compile_options(
            target_runtime,
            precision,
            self.get_graph_output_names(graph_name),
            self.get_graph_channel_last_input(graph_name),
            self.get_graph_channel_last_output(graph_name),
            graph_name,
            other_compile_options,
        )

    def get_hub_link_options(
        self,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
    ) -> str:
        assert target_runtime.is_aot_compiled, (
            "Multi-graph models only support AOT-compiled target runtimes."
        )
        return build_link_options(target_runtime, other_link_options)

    def get_graph_hub_profile_options(
        self,
        graph_name: str,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> str:
        assert target_runtime.is_aot_compiled, (
            "Multi-graph models only support AOT-compiled target runtimes."
        )
        return build_profile_options(target_runtime, graph_name, other_profile_options)

    # -- Auto-built from per-graph getters --

    def get_graph_output_names(self, graph_name: str) -> list[str]:
        outputs = self.get_graph_output_spec(graph_name).keys()
        assert outputs, f"get_output_spec() is not defined oforn {graph_name}!"
        return list(outputs)

    def get_input_spec(self, *args: Any, **kwargs: Any) -> MultiGraphGroup[InputSpec]:
        return MultiGraphGroup(
            {
                name: self.get_graph_input_spec(name, *args, **kwargs)
                for name in self.graph_names
            }
        )

    def get_output_spec(self) -> MultiGraphGroup[OutputSpec]:
        return MultiGraphGroup(
            {name: self.get_graph_output_spec(name) for name in self.graph_names}
        )

    def get_output_names(self) -> MultiGraphGroup[list[str]]:
        return MultiGraphGroup(
            {name: self.get_graph_output_names(name) for name in self.graph_names}
        )

    def get_channel_last_inputs(self) -> MultiGraphGroup[list[str]]:
        return MultiGraphGroup(
            {name: self.get_graph_channel_last_input(name) for name in self.graph_names}
        )

    def get_channel_last_outputs(self) -> MultiGraphGroup[list[str]]:
        return MultiGraphGroup(
            {
                name: self.get_graph_channel_last_output(name)
                for name in self.graph_names
            }
        )

    # -- Auto-built hub option dicts --

    def get_hub_quantize_options(
        self,
        precision: Precision,
        other_quantize_options: str = "",
    ) -> MultiGraphGroup[str]:
        return MultiGraphGroup(
            {
                name: self.get_graph_hub_quantize_options(
                    name, precision, other_quantize_options
                )
                for name in self.graph_names
            }
        )

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> MultiGraphGroup[str]:
        return MultiGraphGroup(
            {
                name: self.get_graph_hub_compile_options(
                    name, target_runtime, precision, other_compile_options, device
                )
                for name in self.graph_names
            }
        )

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> MultiGraphGroup[str]:
        return MultiGraphGroup(
            {
                name: self.get_graph_hub_profile_options(
                    name, target_runtime, other_profile_options
                )
                for name in self.graph_names
            }
        )

    def sample_inputs(
        self,
        input_spec: MultiGraphGroup[InputSpec] | None = None,
        use_channel_last_format: bool = True,
        **kwargs: Any,
    ) -> MultiGraphGroup[SampleInputsType]:
        input_specs = input_spec or self.get_input_spec(**kwargs)
        return MultiGraphGroup(
            {
                graph_name: self.get_graph_sample_inputs(
                    graph_name, spec, use_channel_last_format
                )
                for graph_name, spec in input_specs.items()
            }
        )


class MultiGraphCollectionModel(
    CollectionModel[BaseModel | MultiGraphWorkbenchModel], FromPretrainedProtocol
):
    """A collection model where one or more components have multiple graphs."""

    COMPONENT_BASE_TYPES = (BaseModel, MultiGraphWorkbenchModel)

    @staticmethod
    def eval_datasets() -> list[str]:
        return []

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type]:
        """Returns list of dataset classes on which this model can be evaluated."""
        return []

    def convert_to_hub_source_model(
        self,
        component_name: str,
        graph_name: str | None,
        output_path: str | Path,
        target_runtime: TargetRuntime,
        input_spec: InputSpec | None = None,
    ) -> Path | str:
        component = self.components[component_name]
        result: Path | str | None
        if isinstance(component, MultiGraphWorkbenchModel):
            assert graph_name is not None
            result = component.convert_graph_to_hub_source_model(
                graph_name, output_path, input_spec
            )
        else:
            result = component.convert_to_hub_source_model(
                target_runtime, output_path, input_spec
            )
        assert result is not None
        return result

    # -- Per-component-graph getters --

    def get_component_graph_hub_quantize_options(
        self,
        component_name: str,
        graph_name: str | None,
        precision: Precision,
        other_quantize_options: str = "",
    ) -> str:
        component = self.components[component_name]
        if isinstance(component, MultiGraphWorkbenchModel):
            assert graph_name is not None
            return component.get_graph_hub_quantize_options(
                graph_name, precision, other_quantize_options
            )
        return component.get_hub_quantize_options(precision, other_quantize_options)

    def get_component_graph_hub_compile_options(
        self,
        component_name: str,
        graph_name: str | None,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        component = self.components[component_name]
        if isinstance(component, MultiGraphWorkbenchModel):
            assert graph_name is not None
            return component.get_graph_hub_compile_options(
                graph_name, target_runtime, precision, other_compile_options, device
            )
        return component.get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, component_name
        )

    def get_component_graph_hub_link_options(
        self,
        component_name: str,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
    ) -> str:
        return self.components[component_name].get_hub_link_options(
            target_runtime, other_link_options
        )

    def get_component_graph_hub_profile_options(
        self,
        component_name: str,
        graph_name: str | None,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> str:
        component = self.components[component_name]
        if isinstance(component, MultiGraphWorkbenchModel):
            assert graph_name is not None
            return component.get_graph_hub_profile_options(
                graph_name, target_runtime, other_profile_options
            )
        return component.get_hub_profile_options(
            target_runtime, other_profile_options, component_name
        )

    # -- All-component getters --

    def get_input_spec(
        self,
        per_component_kwargs: ComponentGroup[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> MultiGraphComponentGroup[InputSpec]:
        filtered = self._filter_kwargs_for_each_component(
            "get_input_spec", per_component_kwargs, kwargs
        )
        out: MultiGraphComponentGroup[InputSpec] = MultiGraphComponentGroup()
        for comp_name, component in self.components.items():
            comp_kwargs = filtered.get(comp_name, {})
            if isinstance(component, MultiGraphWorkbenchModel):
                for graph_name, spec in component.get_input_spec(**comp_kwargs).items():
                    out[(comp_name, graph_name)] = spec
            else:
                out[(comp_name, None)] = component.get_input_spec(**comp_kwargs)
        return out

    def get_unsupported_reason(
        self,
        target_runtime: TargetRuntime,
        device: Device,
    ) -> str | None:
        for comp_name, component in self.components.items():
            if reason := component.get_unsupported_reason(target_runtime, device):
                return f"Component {comp_name}: {reason}"
        return None

    def get_hub_quantize_options(
        self,
        precision: Precision,
        other_quantize_options: str = "",
        per_component_quantize_options: ComponentGroup[str] | None = None,
    ) -> MultiGraphComponentGroup[str]:
        per_component_quantize_options = (
            per_component_quantize_options or ComponentGroup()
        )
        out: MultiGraphComponentGroup[str] = MultiGraphComponentGroup()
        for comp_name, component in self.components.items():
            comp_opts = (
                other_quantize_options
                + f" {per_component_quantize_options.get(comp_name, '')}"
            )
            if isinstance(component, MultiGraphWorkbenchModel):
                for graph_name, opts in component.get_hub_quantize_options(
                    precision, comp_opts
                ).items():
                    out[(comp_name, graph_name)] = opts
            else:
                out[(comp_name, None)] = component.get_hub_quantize_options(
                    precision, comp_opts
                )
        return out

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        per_component_compile_options: ComponentGroup[str] | None = None,
    ) -> MultiGraphComponentGroup[str]:
        per_component_compile_options = (
            per_component_compile_options or ComponentGroup()
        )
        out: MultiGraphComponentGroup[str] = MultiGraphComponentGroup()
        for comp_name, component in self.components.items():
            comp_opts = (
                other_compile_options
                + f" {per_component_compile_options.get(comp_name, '')}"
            )
            if isinstance(component, MultiGraphWorkbenchModel):
                for graph_name, opts in component.get_hub_compile_options(
                    target_runtime, precision, comp_opts, device
                ).items():
                    out[(comp_name, graph_name)] = opts
            else:
                out[(comp_name, None)] = component.get_hub_compile_options(
                    target_runtime, precision, comp_opts, device, comp_name
                )
        return out

    def get_hub_link_options(
        self,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
        per_component_link_options: ComponentGroup[str] | None = None,
    ) -> ComponentGroup[str]:
        per_component_link_options = per_component_link_options or ComponentGroup()
        out: ComponentGroup[str] = ComponentGroup()
        for comp_name, component in self.components.items():
            comp_opts = (
                other_link_options + f" {per_component_link_options.get(comp_name, '')}"
            )
            if isinstance(component, MultiGraphWorkbenchModel):
                out[comp_name] = component.get_hub_link_options(
                    target_runtime, comp_opts
                )
            else:
                out[comp_name] = component.get_hub_link_options(
                    target_runtime, comp_opts
                )
        return out

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        per_component_profile_options: ComponentGroup[str] | None = None,
    ) -> MultiGraphComponentGroup[str]:
        per_component_profile_options = (
            per_component_profile_options or ComponentGroup()
        )
        out: MultiGraphComponentGroup[str] = MultiGraphComponentGroup()
        for comp_name, component in self.components.items():
            comp_opts = (
                other_profile_options
                + f" {per_component_profile_options.get(comp_name, '')}"
            )
            if isinstance(component, MultiGraphWorkbenchModel):
                for graph_name, opts in component.get_hub_profile_options(
                    target_runtime, comp_opts
                ).items():
                    out[(comp_name, graph_name)] = opts
            else:
                out[(comp_name, None)] = component.get_hub_profile_options(
                    target_runtime, comp_opts, comp_name
                )
        return out

    def sample_inputs(
        self,
        input_spec: MultiGraphComponentGroup[InputSpec] | None = None,
        use_channel_last_format: bool = True,
    ) -> MultiGraphComponentGroup[SampleInputsType]:
        input_specs = input_spec or self.get_input_spec()
        out: MultiGraphComponentGroup[SampleInputsType] = MultiGraphComponentGroup()
        for comp_name, component in self.components.items():
            if isinstance(component, MultiGraphWorkbenchModel):
                graph_specs: MultiGraphGroup[InputSpec] = MultiGraphGroup()
                for graph_name in component.graph_names:
                    key = (comp_name, graph_name)
                    if spec := input_specs.get(key):
                        graph_specs[graph_name] = spec
                for graph_name, sample in component.sample_inputs(
                    graph_specs or None, use_channel_last_format
                ).items():
                    out[(comp_name, graph_name)] = sample
            else:
                out[(comp_name, None)] = component.sample_inputs(
                    input_specs.get((comp_name, None)), use_channel_last_format
                )
        return out
