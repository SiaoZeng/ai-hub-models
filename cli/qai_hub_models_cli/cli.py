# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import sys
from collections.abc import Callable, Iterable
from importlib.metadata import PackageNotFoundError, version
from typing import TypeVar

from packaging.version import Version
from packaging.version import parse as parse_version
from prettytable import PrettyTable

from qai_hub_models_cli._internal.utils import is_internal_repo, use_internal_releases
from qai_hub_models_cli._version import __version__
from qai_hub_models_cli.common import (
    AIHUB_MODELS_URL,
    build_filter_command,
    format_command_sections,
    model_repo_url,
    parse_sdk_version_filters,
    sample_command,
)
from qai_hub_models_cli.envvars import (
    VERBOSE_EXCEPTIONS_ENVVAR,
    bool_envvar_value,
)
from qai_hub_models_cli.fetch import fetch, get_asset_url
from qai_hub_models_cli.proto.info_pb2 import MODEL_TAG_LLM, ModelDomain, ModelTag
from qai_hub_models_cli.proto.manifest_pb2 import ManifestModelEntry
from qai_hub_models_cli.proto.platform_pb2 import FormFactor, WebsiteWorld
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.proto_helpers.info import get_model_info
from qai_hub_models_cli.proto_helpers.manifest import get_manifest, get_manifest_entry
from qai_hub_models_cli.proto_helpers.numerics import (
    filter_numerics,
    format_numerics_table,
    get_model_numerics,
)
from qai_hub_models_cli.proto_helpers.perf import (
    filter_perf,
    format_perf_table,
    get_model_perf,
)
from qai_hub_models_cli.proto_helpers.platform import (
    filter_chipsets,
    filter_devices,
    format_chipsets_table,
    format_devices_table,
    format_runtime_links,
    format_runtimes_table,
    format_similar_devices_table,
    get_platform,
    resolve_chipset,
)
from qai_hub_models_cli.proto_helpers.platform_enums import (
    domain_proto_to_str,
    form_factor_proto_to_str,
    form_factor_str_to_proto,
    license_proto_to_str,
    os_str_to_proto,
    precision_proto_to_str,
    runtime_proto_to_str,
    runtime_str_to_proto,
    tag_proto_to_str,
    tag_str_to_proto,
    use_case_proto_to_str,
    world_proto_to_str,
    world_str_to_proto,
)
from qai_hub_models_cli.proto_helpers.release_assets import (
    filter_release_assets,
    format_fetch_commands,
    format_release_assets_table,
    format_tool_versions,
    get_model_asset_details,
    get_model_release_assets,
)
from qai_hub_models_cli.utils import build_table, wrap_table_column
from qai_hub_models_cli.versions import (
    CURRENT_VERSION,
    MIN_MODEL_FILTER_VERSION,
    UnsupportedVersionError,
    get_supported_versions,
    normalize_version,
    print_upgrade_notice,
    version_flag,
)

T = TypeVar("T")


def _check_version_match() -> None:
    """Exit if qai_hub_models and qai_hub_models_cli versions differ."""
    try:
        cli_version = version("qai_hub_models_cli")
        models_version = version("qai_hub_models")
    except PackageNotFoundError:
        return
    if cli_version != models_version:
        print(
            f"Version mismatch: qai_hub_models_cli=={cli_version} "
            f"but qai_hub_models=={models_version}. "
            "Please reinstall both packages from the same version."
        )
        sys.exit(1)


def _parse_version(s: str) -> Version:
    """Argparse type function: normalize and parse a version string."""
    return parse_version(normalize_version(s))


def _add_version_arg(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``-v/--version`` argument (stored as ``qaihm_version``)."""
    parser.add_argument(
        "-v",
        "--version",
        default=CURRENT_VERSION,
        type=_parse_version,
        dest="qaihm_version",
        help=f"AI Hub Models version tag (e.g. v0.45.0 or 0.45.0). Default: {__version__}.",
    )


def _add_quiet_arg(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add the shared ``-q/--quiet`` flag with a command-specific help string."""
    parser.add_argument("-q", "--quiet", action="store_true", help=help_text)


def _flatten_multi_arg(value: list[list[T]] | None) -> list[T] | None:
    """Flatten an ``action="append", nargs="+"`` value into a single list.

    Such args parse to a list-of-lists (one inner list per flag occurrence), so
    both ``--flag a b`` and repeated ``--flag a --flag b`` accumulate rather than
    overwrite. Returns None when the flag was not given.
    """
    if not value:
        return None
    return [item for group in value for item in group]


def _add_chipset_attribute_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add the chipset-attribute filters shared by the devices/chipsets commands."""
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Only show entries whose chipset supports fp16.",
    )
    parser.add_argument(
        "--htp-version",
        nargs="+",
        action="append",
        type=int,
        default=None,
        help="Filter by chipset HTP version(s).",
    )
    parser.add_argument(
        "--soc-model",
        nargs="+",
        action="append",
        type=int,
        default=None,
        help="Filter by chipset SoC model(s).",
    )


def _run_fetch(args: argparse.Namespace) -> None:
    sdk_versions = parse_sdk_version_filters(args.sdk_version or [])

    if args.info:
        all_assets = get_model_release_assets(args.model, args.qaihm_version)
        platform = get_platform(args.qaihm_version)
        release_assets = filter_release_assets(
            all_assets,
            platform,
            args.runtime,
            args.precision,
            args.chipset,
            args.device,
            sdk_versions,
        )
        if not release_assets.assets:
            print("No release assets match the given filters.")
            return
        print(
            format_release_assets_table(
                release_assets,
                platform.chipsets,
                title="Download Options",
            )
        )
        print()
        print(
            format_fetch_commands(
                release_assets,
                args.model,
                # The user is already running -i, so don't suggest it again.
                subset=False,
                runtime=args.runtime,
                precision=args.precision,
                chipset=args.chipset,
                device=args.device,
                sdk_versions=sdk_versions,
                version=args.qaihm_version,
            )
        )
        return

    try:
        if args.url_only:
            url = get_asset_url(
                model=args.model,
                runtime=args.runtime,
                precision=args.precision,
                version=args.qaihm_version,
                chipset=args.chipset,
                device=args.device,
                quiet=args.quiet,
                url_only=True,
                sdk_versions=sdk_versions,
            )
            print(url)
            return

        result = fetch(
            model=args.model,
            runtime=args.runtime,
            precision=args.precision,
            chipset=args.chipset,
            device=args.device,
            version=args.qaihm_version,
            extract=args.extract,
            output_dir=args.output_dir,
            quiet=args.quiet,
            sdk_versions=sdk_versions,
        )
    except Exception as e:
        if args.quiet and not isinstance(
            e, (FileNotFoundError, UnsupportedVersionError)
        ):
            print(
                "Failed to fetch model. Consider excluding -q/--quiet from your command to reveal more logs."
            )
        raise

    if args.quiet:
        print(result)
        return

    if args.extract:
        print(f"Extracted to: {result}")
    else:
        print(f"Saved to: {result}")

    try:
        asset = get_model_asset_details(
            get_model_release_assets(args.model, args.qaihm_version),
            get_platform(args.qaihm_version),
            args.runtime,
            args.precision,
            args.chipset,
            args.device,
        )
    except Exception:
        asset = None
    if asset is not None and asset.HasField("tool_versions"):
        print(
            f"\nThis download was verified with: "
            f"{format_tool_versions(asset.tool_versions)}\n"
            "Run the model with matching versions to match our reported numerics and performance. Other "
            "versions may behave differently or fail to run."
        )

    print_upgrade_notice()


def add_fetch_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "fetch",
        help="Download a pre-compiled model asset.",
    )
    parser.add_argument("model", type=str.lower, help="Model ID (e.g. mobilenet_v2).")
    runtime_values = ", ".join(
        [
            runtime_proto_to_str(r)
            for r in Runtime.values()
            if r != Runtime.RUNTIME_UNSPECIFIED
        ]
    )
    parser.add_argument(
        "-r",
        "--runtime",
        default=None,
        help=f"Target runtime. Known values: {runtime_values}. "
        "Older releases may support different values. "
        "Required unless -i/--info is given.",
    )
    precision_values = ", ".join(
        [
            precision_proto_to_str(p)
            for p in Precision.values()
            if p != Precision.PRECISION_UNSPECIFIED
        ]
    )
    parser.add_argument(
        "-p",
        "--precision",
        default=None,
        type=str.lower,
        help=f"Model precision. Known values: {precision_values}. "
        "Older releases may support different values.",
    )
    # TODO(#18389): Add a list of valid chipsets
    # so the CLI can validate and suggest chipset names.
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "-c",
        "--chipset",
        default=None,
        type=str.lower,
        help="Chipset name for device-specific (AOT compiled) runtimes. "
        "Run `qai-hub-models chipsets` to see supported chipsets.",
    )
    target.add_argument(
        "-d",
        "--device",
        default=None,
        help="Device name for device-specific (AOT compiled) runtimes. "
        "Run `qai-hub-models devices` to see supported devices. Cannot be specified with chipset.",
    )
    parser.add_argument(
        "-s",
        "--sdk-version",
        nargs="+",
        default=None,
        type=str.lower,
        help="Filter by SDK/tool version using 'tool=version' syntax (e.g. "
        "'litert=1.4.4' or 'qairt=2.20'). Accepts multiple values; an asset "
        "must match all of them.",
    )
    _add_version_arg(parser)
    parser.add_argument(
        "--extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract the downloaded zip archive (default: true). Use --no-extract to skip.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Output directory. Default: current directory.",
    )
    parser.add_argument(
        "--url-only",
        action="store_true",
        help="Print the download URL only (do not download).",
    )
    parser.add_argument(
        "-i",
        "--info",
        action="store_true",
        help="List the supported release assets and return without downloading. "
        "The runtime, precision, chipset, device, and sdk-version args act as filters.",
    )
    _add_quiet_arg(parser, "Suppress all output except the result path.")
    parser.set_defaults(func=_run_fetch)
    return parser


def _add_model_metric_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add the runtime/precision/chipset/device/sdk filters shared by the
    ``perf`` and ``numerics`` commands. Mirrors the ``fetch`` filter flags.
    """
    runtime_values = ", ".join(
        runtime_proto_to_str(r)
        for r in Runtime.values()
        if r != Runtime.RUNTIME_UNSPECIFIED
    )
    parser.add_argument(
        "-r",
        "--runtime",
        nargs="+",
        action="append",
        default=None,
        help="Filter by runtime(s); a record matches any of them. "
        "May be repeated or given multiple values. "
        f"Known values: {runtime_values}.",
    )
    precision_values = ", ".join(
        precision_proto_to_str(p)
        for p in Precision.values()
        if p != Precision.PRECISION_UNSPECIFIED
    )
    parser.add_argument(
        "-p",
        "--precision",
        nargs="+",
        action="append",
        default=None,
        type=str.lower,
        help="Filter by precision(s); a record matches any of them. "
        "May be repeated or given multiple values. "
        f"Known values: {precision_values}.",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "-c",
        "--chipset",
        nargs="+",
        action="append",
        default=None,
        type=str.lower,
        help="Filter by chipset(s); a record matches any of them. "
        "May be repeated or given multiple values. "
        "Run `qai-hub-models chipsets` to see supported chipsets.",
    )
    target.add_argument(
        "-d",
        "--device",
        nargs="+",
        action="append",
        default=None,
        help="Filter by device(s); a record matches any of them. "
        "May be repeated or given multiple values. "
        "Run `qai-hub-models devices` to see supported devices. "
        "Cannot be combined with --chipset.",
    )


def _print_model_metric_footer(command: str, args: argparse.Namespace) -> None:
    """Print the related-command hints shown beneath a perf/numerics table.

    Points at the listing commands for the dimensions a user can filter on, plus
    an example ``command`` invocation pre-filled with the filters already passed
    (placeholders for the rest).
    """
    vflag = version_flag(args.qaihm_version)
    filter_cmd = build_filter_command(
        command,
        args.model,
        vflag,
        runtimes=_flatten_multi_arg(args.runtime),
        precisions=_flatten_multi_arg(args.precision),
        chipsets=_flatten_multi_arg(args.chipset),
        devices=_flatten_multi_arg(args.device),
    )
    # The component and sdk-version filters are perf-only.
    for comp in _flatten_multi_arg(getattr(args, "component", None)) or []:
        filter_cmd += f" --component '{comp}'"
    for query in getattr(args, "sdk_version", None) or []:
        filter_cmd += f" -s '{query}'"

    # Cross-link to the sibling metric command (perf <-> numerics).
    sibling = "numerics" if command == "perf" else "perf"
    sibling_label = (
        "Accuracy metrics" if sibling == "numerics" else "Performance metrics"
    )

    print()
    print(
        format_command_sections(
            {
                "Platform Info": [
                    ("More about runtimes", sample_command("runtimes", vflag)),
                    ("Chipset information", sample_command("chipsets", vflag)),
                    ("See devices per chipset", sample_command("devices", vflag)),
                ],
                "Model Info": [
                    ("Full model details", sample_command("info", args.model, vflag)),
                    (sibling_label, sample_command(sibling, args.model, vflag)),
                    ("Filter these results", filter_cmd),
                ],
            }
        )
    )


def _run_perf(args: argparse.Namespace) -> None:
    sdk_versions = parse_sdk_version_filters(args.sdk_version or [])
    perf = get_model_perf(args.model, args.qaihm_version)
    perf = filter_perf(
        perf,
        get_platform(args.qaihm_version),
        runtime=_flatten_multi_arg(args.runtime),
        precision=_flatten_multi_arg(args.precision),
        chipset=_flatten_multi_arg(args.chipset),
        device=_flatten_multi_arg(args.device),
        sdk_versions=sdk_versions,
        components=_flatten_multi_arg(args.component),
    )
    print(format_perf_table(perf))
    if perf.performance_metrics:
        _print_model_metric_footer("perf", args)


def add_perf_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "perf",
        help="Show a model's performance metrics.",
        description="Display per-device performance metrics (inference time, "
        "memory, compute unit) for a model. The runtime, precision, chipset, "
        "device, and sdk-version args act as filters.",
    )
    parser.add_argument(
        "model", type=str.lower, help="Model ID or display name (e.g. mobilenet_v2)."
    )
    _add_model_metric_filter_args(parser)
    parser.add_argument(
        "-s",
        "--sdk-version",
        nargs="+",
        default=None,
        type=str.lower,
        help="Filter by SDK/tool version using 'tool=version' syntax (e.g. "
        "'litert=1.4.4' or 'qairt=2.20'). Accepts multiple values; a record "
        "must match all of them.",
    )
    parser.add_argument(
        "--component",
        nargs="+",
        action="append",
        default=None,
        help="Filter to the given component(s), for multi-component models. "
        "May be repeated or given multiple values.",
    )
    _add_version_arg(parser)
    parser.set_defaults(func=_run_perf)
    return parser


def _run_numerics(args: argparse.Namespace) -> None:
    numerics = get_model_numerics(args.model, args.qaihm_version)
    numerics = filter_numerics(
        numerics,
        get_platform(args.qaihm_version),
        runtime=_flatten_multi_arg(args.runtime),
        precision=_flatten_multi_arg(args.precision),
        chipset=_flatten_multi_arg(args.chipset),
        device=_flatten_multi_arg(args.device),
    )
    print(format_numerics_table(numerics))
    if numerics.metrics:
        _print_model_metric_footer("numerics", args)


def add_numerics_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "numerics",
        help="Show a model's accuracy metrics.",
        description="Display per-device numerical accuracy metrics for a model, "
        "alongside the torch reference value. The runtime, precision, chipset, "
        "and device args act as filters.",
    )
    parser.add_argument(
        "model", type=str.lower, help="Model ID or display name (e.g. mobilenet_v2)."
    )
    _add_model_metric_filter_args(parser)
    _add_version_arg(parser)
    parser.set_defaults(func=_run_numerics)
    return parser


def _run_list_models(args: argparse.Namespace) -> None:
    manifest = get_manifest(args.qaihm_version)

    # All filters except --domain rely on manifest fields added in
    # MIN_MODEL_FILTER_VERSION; older releases only support --domain.
    gated_filters = (
        args.quantized,
        args.runtime,
        args.aot,
        args.jit,
        args.tag,
        args.chipset,
        args.device,
        args.llm,
    )
    if any(gated_filters) and args.qaihm_version < MIN_MODEL_FILTER_VERSION:
        print(
            f"Filtering by quantization, runtime, chipset, device, or tag requires "
            f"version {MIN_MODEL_FILTER_VERSION} or later. Only --domain is supported "
            f"for version {args.qaihm_version}."
        )
        return

    # Resolve/validate each filter's criteria once up front, then build a list of
    # per-model predicates so the models are walked a single time below.
    predicates: list[Callable[[ManifestModelEntry], bool]] = []

    if args.domain:

        def _normalize_domain(s: str) -> str:
            return s.lower().replace("_", " ").replace("-", " ")

        domain_filter = _normalize_domain(args.domain)
        predicates.append(
            lambda e: _normalize_domain(domain_proto_to_str(e.domain)) == domain_filter
        )

    if args.quantized:
        predicates.append(lambda e: e.is_quantized)

    if args.aot or args.jit or args.runtime:
        platform_runtimes = get_platform(args.qaihm_version).runtimes
        if runtimes := _flatten_multi_arg(args.runtime):
            try:
                runtime_vals = {runtime_str_to_proto(r) for r in runtimes}
            except KeyError as e:
                print(str(e))
                return
            predicates.append(lambda e: runtime_vals.issubset(e.supported_runtimes))
        if args.aot:
            aot = {rt.runtime for rt in platform_runtimes if rt.is_aot_compiled}
            predicates.append(lambda e: bool(aot.intersection(e.supported_runtimes)))
        if args.jit:
            jit = {rt.runtime for rt in platform_runtimes if not rt.is_aot_compiled}
            predicates.append(lambda e: bool(jit.intersection(e.supported_runtimes)))

    if tags := _flatten_multi_arg(args.tag):
        try:
            tag_vals = {tag_str_to_proto(t) for t in tags}
        except KeyError as e:
            print(str(e))
            return
        predicates.append(lambda e: tag_vals.issubset(e.tags))

    if args.llm:
        predicates.append(lambda e: MODEL_TAG_LLM in e.tags)

    if args.chipset or args.device:
        try:
            chipset_name = resolve_chipset(
                get_platform(args.qaihm_version),
                chipset=args.chipset,
                device=args.device,
            ).name
        except KeyError as e:
            print(str(e))
            return
        predicates.append(lambda e: chipset_name in e.supported_chipsets)

    entries = sorted(
        (e for e in manifest.models if all(p(e) for p in predicates)),
        key=lambda e: e.id,
    )

    if not entries:
        print("No models found.")
        return

    if args.quiet:
        for entry in entries:
            print(entry.id)
        return

    groups: dict[str, list[ManifestModelEntry]] = {}
    for entry in entries:
        domain = domain_proto_to_str(entry.domain)
        groups.setdefault(domain, []).append(entry)

    # The Quantized/Runtimes columns are populated from manifest fields added in
    # MIN_MODEL_FILTER_VERSION; on older releases they'd be blank, so omit them.
    show_filter_columns = args.qaihm_version >= MIN_MODEL_FILTER_VERSION
    if show_filter_columns:
        columns = ["Name", "Domain", "Quantized", "Runtimes"]
        wrap_column, wrap_on_commas = "Runtimes", True
    else:
        columns = ["Name", "Domain"]
        wrap_column, wrap_on_commas = "Name", False

    rows = []
    for domain, group in groups.items():
        for entry in group:
            row = [entry.display_name, domain]
            if show_filter_columns:
                row += [
                    "Yes" if entry.is_quantized else "No",
                    ", ".join(
                        runtime_proto_to_str(r) for r in entry.supported_runtimes
                    ),
                ]
            rows.append(row)

    print(
        build_table(
            columns,
            rows,
            wrap_column=wrap_column,
            wrap_on_commas=wrap_on_commas,
            title="Models",
        )
    )

    print(f"Total: {len(entries)} models\n")
    print("Looking for something else?")
    print(
        " - Use AI Hub Workbench to bring your own model: https://aihub.qualcomm.com/get-started#workbench"
    )
    print(
        " - Request we add a new model: https://github.com/qualcomm/ai-hub-models/issues\n"
    )
    print(
        f"More about our supported platforms: `{sample_command('runtimes')}`, "
        f"`{sample_command('devices')}`, `{sample_command('chipsets')}`\n"
    )
    print(
        f"Run `{sample_command('info', '<model_id>')}` for details and download options."
    )
    print_upgrade_notice()


def add_list_models_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "models",
        help="List all available models.",
        description="List all models available in a given AI Hub Models release.",
    )
    _add_version_arg(parser)
    domain_values = ", ".join(
        domain_proto_to_str(d)
        for d in ModelDomain.values()
        if d != ModelDomain.MODEL_DOMAIN_UNSPECIFIED
    )
    parser.add_argument(
        "--domain",
        default=None,
        type=str.lower,
        help=f"Filter by domain. Known values: {domain_values}.",
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Filter to quantized models.",
    )
    runtime_values = ", ".join(
        runtime_proto_to_str(r)
        for r in Runtime.values()
        if r != Runtime.RUNTIME_UNSPECIFIED
    )
    parser.add_argument(
        "-r",
        "--runtime",
        nargs="+",
        action="append",
        default=None,
        type=str.lower,
        help="Filter to models with assets for all of the given runtimes. "
        "May be repeated or given multiple values. "
        f"Known values: {runtime_values}.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Filter to Large Language Models and Vision Language Models.",
    )
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--aot",
        action="store_true",
        help="Filter to models with ahead-of-time (device-specific) compiled assets.",
    )
    compile_group.add_argument(
        "--jit",
        action="store_true",
        help="Filter to models with just-in-time (universal) compiled assets.",
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "-c",
        "--chipset",
        default=None,
        type=str.lower,
        help="Filter by a chipset the model has been profiled on. "
        "Run `qai-hub-models chipsets` to see supported chipsets.",
    )
    target_group.add_argument(
        "-d",
        "--device",
        default=None,
        help="Filter by a device the model has been profiled on. "
        "Run `qai-hub-models devices` to see supported devices. "
        "Cannot be combined with --chipset.",
    )
    tag_values = ", ".join(
        tag_proto_to_str(t)
        for t in ModelTag.values()
        if t != ModelTag.MODEL_TAG_UNSPECIFIED
    )
    parser.add_argument(
        "-t",
        "--tag",
        nargs="+",
        action="append",
        default=None,
        type=str.lower,
        help="Filter to models with all of the given tags. "
        "May be repeated or given multiple values. "
        f"Known values: {tag_values}.",
    )
    _add_quiet_arg(parser, "Print model IDs only, one per line.")
    parser.set_defaults(func=_run_list_models)
    return parser


def _run_list_devices(args: argparse.Namespace) -> None:
    platform = get_platform(args.qaihm_version)
    devices = sorted(
        platform.devices,
        key=lambda d: (form_factor_proto_to_str(d.form_factor), d.name),
    )
    types = _flatten_multi_arg(args.type)
    oses = _flatten_multi_arg(args.os)
    devices = filter_devices(
        devices,
        platform.chipsets,
        form_factor=[form_factor_str_to_proto(t) for t in types] if types else None,
        os=[os_str_to_proto(o) for o in oses] if oses else None,
        fp16=True if args.fp16 else None,
        htp_version=_flatten_multi_arg(args.htp_version),
        soc_model=_flatten_multi_arg(args.soc_model),
    )

    if not devices:
        print("No devices found.")
        return

    if args.quiet:
        for device in devices:
            print(device.name)
        return

    # Devices with a reference_chipset are "similar" devices whose perf numbers
    # are duplicated from another chipset; show them in a separate table.
    primary = [d for d in devices if not d.reference_chipset]
    similar = [d for d in devices if d.reference_chipset]

    print(format_devices_table(primary, platform.chipsets))
    print(f"Total: {len(primary)} devices.")

    if similar:
        print()
        print(format_similar_devices_table(similar, platform.chipsets))
        print(
            f"Total: {len(similar)} similar devices. NOTE: The similar devices table lists devices that have not "
            "been tested with AI Hub Models. However, the corresponding similar device / chipset "
            "serve as substitute compilation targets and have been tested. Assets built for the 'similar device' / 'similar chipset' "
            "are likely to run on the device, though performance and accuracy metrics may differ."
        )

    print(
        f"\nNOTE: This is a snapshot of devices tested with AI Hub Models v{args.qaihm_version}. AI Hub Workbench may support a different set of devices."
    )

    print("\nSee all supported chipsets using `qai-hub-models chipsets`.")

    print_upgrade_notice()


def add_list_devices_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "devices",
        help="List all supported devices.",
        description="List all devices supported in a given AI Hub Models release.",
    )
    _add_version_arg(parser)
    type_values = ", ".join(
        form_factor_proto_to_str(f)
        for f in FormFactor.values()
        if f != FormFactor.FORM_FACTOR_UNSPECIFIED
    )
    parser.add_argument(
        "-t",
        "--type",
        nargs="+",
        action="append",
        default=None,
        help=f"Filter by device type(s). Known values: {type_values}.",
    )
    parser.add_argument(
        "--os",
        nargs="+",
        action="append",
        default=None,
        help="Filter by operating system(s) (e.g. Android, Windows).",
    )
    _add_chipset_attribute_filter_args(parser)
    _add_quiet_arg(parser, "Print device names only, one per line.")
    parser.set_defaults(func=_run_list_devices)
    return parser


def _run_list_chipsets(args: argparse.Namespace) -> None:
    chipsets = sorted(
        get_platform(args.qaihm_version).chipsets,
        key=lambda c: (world_proto_to_str(c.world), c.marketing_name),
    )
    types = _flatten_multi_arg(args.type)
    chipsets = filter_chipsets(
        chipsets,
        world=[world_str_to_proto(t) for t in types] if types else None,
        fp16=True if args.fp16 else None,
        htp_version=_flatten_multi_arg(args.htp_version),
        soc_model=_flatten_multi_arg(args.soc_model),
    )

    if not chipsets:
        print("No chipsets found.")
        return

    if args.quiet:
        for chipset in chipsets:
            print(chipset.marketing_name)
        return

    print(format_chipsets_table(chipsets))

    print(f"Total: {len(chipsets)} chipsets")
    print(
        f"\nNOTE: This is a snapshot of chipsets tested with AI Hub Models v{args.qaihm_version}. AI Hub Workbench may support a different set of devices."
    )
    print("\nSee all supported devices using `qai-hub-models devices`.")
    print_upgrade_notice()


def add_list_chipsets_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "chipsets",
        help="List all supported chipsets.",
        description="List all chipsets supported in a given AI Hub Models release.",
    )
    _add_version_arg(parser)
    type_values = ", ".join(
        world_proto_to_str(w)
        for w in WebsiteWorld.values()
        if w != WebsiteWorld.WEBSITE_WORLD_UNSPECIFIED
    )
    parser.add_argument(
        "-t",
        "--type",
        nargs="+",
        action="append",
        default=None,
        help=f"Filter by chipset type(s). Known values: {type_values}.",
    )
    _add_chipset_attribute_filter_args(parser)
    _add_quiet_arg(parser, "Print chipset names only, one per line.")
    parser.set_defaults(func=_run_list_chipsets)
    return parser


def _run_list_runtimes(args: argparse.Namespace) -> None:
    runtimes = get_platform(args.qaihm_version).runtimes

    if args.quiet:
        for rt in runtimes:
            print(runtime_proto_to_str(rt.runtime))
        return

    print(format_runtimes_table(runtimes, args.qaihm_version))
    print(f"Total: {len(runtimes)} runtimes")
    # Display metadata (incl. docs links) exists only as of MIN_MODEL_FILTER_VERSION.
    if args.qaihm_version >= MIN_MODEL_FILTER_VERSION and (
        links := format_runtime_links(runtimes)
    ):
        print(f"\n{links}")
    print_upgrade_notice()


def add_list_runtimes_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "runtimes",
        help="List all runtimes.",
        description="List all runtimes models can be compiled to.",
    )
    _add_version_arg(parser)
    _add_quiet_arg(parser, "Print runtime IDs only, one per line.")
    parser.set_defaults(func=_run_list_runtimes)
    return parser


def _run_info(args: argparse.Namespace) -> None:
    info = get_model_info(args.model, args.qaihm_version)

    url = f"{AIHUB_MODELS_URL}/{info.id}"
    width = max(len(info.name), len(url)) + 4
    print("+" + "=" * width + "+")
    print(f"| {info.name:^{width - 2}} |")
    print(f"| {url:^{width - 2}} |")
    print("+" + "=" * width + "+")
    print()

    if info.description:
        desc_table = PrettyTable()
        desc_table.title = "Description"
        desc_table.header = False
        desc_table.align = "l"
        desc_table.add_row([info.description])
        wrap_table_column(desc_table, 0)
        print(desc_table)
        print()

    metadata_table = PrettyTable()
    metadata_table.header = False
    metadata_table.align = "l"
    if info.domain:
        metadata_table.add_row(["Domain", domain_proto_to_str(info.domain)])
    if info.use_case:
        metadata_table.add_row(["Use Case", use_case_proto_to_str(info.use_case)])
    if info.tags:
        metadata_table.add_row(
            ["Tags", ", ".join(tag_proto_to_str(t) for t in info.tags)]
        )
    # is_quantized / supported_runtimes are manifest fields added in 0.56.0.
    # Best-effort: skip if the model isn't in the manifest for this version.
    if args.qaihm_version >= MIN_MODEL_FILTER_VERSION:
        try:
            entry = get_manifest_entry(args.model, args.qaihm_version)
        except KeyError:
            entry = None
        if entry is not None:
            metadata_table.add_row(["Quantized", "Yes" if entry.is_quantized else "No"])
            if entry.supported_runtimes:
                metadata_table.add_row(
                    [
                        "Supported Runtimes",
                        ", ".join(
                            runtime_proto_to_str(r) for r in entry.supported_runtimes
                        ),
                    ]
                )
            if entry.supported_chipsets:
                chipset_names = {
                    c.name: c.marketing_name
                    for c in get_platform(args.qaihm_version).chipsets
                }
                metadata_table.add_row(
                    [
                        "Supported Chipsets",
                        ", ".join(
                            chipset_names.get(c, c) for c in entry.supported_chipsets
                        ),
                    ]
                )
    if info.license_type:
        license_str = license_proto_to_str(info.license_type)
        if info.HasField("license_url"):
            license_str += f" ({info.license_url})"
        metadata_table.add_row(["License", license_str])
    if info.HasField("source_repo"):
        metadata_table.add_row(["Source Repo", info.source_repo])
    if info.HasField("research_paper"):
        title = (
            info.research_paper_title
            if info.HasField("research_paper_title")
            else "Paper"
        )
        metadata_table.add_row(["Paper", f"{title} ({info.research_paper})"])
    if metadata_table.rows:
        metadata_table.title = "Metadata"
        wrap_table_column(metadata_table, 1)
        print(metadata_table)
        print()

    def _technical_details_table(title: str, details: Iterable) -> PrettyTable:
        table = PrettyTable()
        table.title = title
        table.header = False
        table.align = "l"
        for detail in details:
            if detail.HasField("string_value"):
                val = detail.string_value
            elif detail.HasField("int_value"):
                val = str(detail.int_value)
            elif detail.HasField("float_value"):
                val = str(detail.float_value)
            else:
                val = ""
            table.add_row([detail.key, val])
        wrap_table_column(table, 1)
        return table

    if info.technical_details:
        print(_technical_details_table("Technical Details", info.technical_details))
        print()

    for rt_details in info.runtime_technical_details:
        runtime_name = runtime_proto_to_str(
            rt_details.runtime, get_platform(args.qaihm_version), display_name=True
        )
        print(
            _technical_details_table(
                f"Technical Details ({runtime_name})",
                rt_details.technical_details,
            )
        )
        print()

    try:
        release_assets = get_model_release_assets(args.model, args.qaihm_version)
        print(
            format_release_assets_table(
                release_assets,
                get_platform(args.qaihm_version).chipsets,
                title="Download Options",
            )
        )
        print()
        print(
            format_fetch_commands(
                release_assets,
                args.model,
                include_metrics=True,
                version=args.qaihm_version,
            )
        )
        print()
        print(
            f"Most models can be further customized beyond what is offered by standard model downloads. Scripts that can export the model from source are available at {model_repo_url(info.id, args.qaihm_version)}"
        )
    except (FileNotFoundError, UnsupportedVersionError) as e:
        err_table = PrettyTable()
        err_table.title = "Download Options"
        err_table.header = False
        err_table.align = "l"
        err_table.add_row([str(e)])
        wrap_table_column(err_table, 0)
        print(err_table)


def add_info_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "info",
        help="Show detailed information about a model.",
        description="Display model metadata including description, license, "
        "technical details, and available download options.",
    )
    parser.add_argument(
        "model",
        type=str.lower,
        help="Model ID or display name (e.g. mobilenet_v2).",
    )
    _add_version_arg(parser)
    parser.set_defaults(func=_run_info)
    return parser


def _run_versions(args: argparse.Namespace) -> None:
    supported = get_supported_versions(force_refresh=True)
    installed = CURRENT_VERSION

    if args.quiet:
        print(", ".join(str(v) for v in supported))
        return

    print("Supported AI Hub Models Versions:")
    labeled = [f"{v} (installed)" if v == installed else str(v) for v in supported]
    print(", ".join(labeled))
    print_upgrade_notice()


def add_versions_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "versions",
        help="List all AI Hub Models versions supported by this CLI.",
        description="List all AI Hub Models versions supported by this CLI. "
        "Shows which version is currently installed and whether newer versions are available.",
    )
    _add_quiet_arg(
        parser,
        "Print versions as a plain comma-separated list without the (installed) marker or upgrade notice.",
    )
    parser.set_defaults(func=_run_versions)
    return parser


def _run_validate_aws(args: argparse.Namespace) -> None:
    from qai_hub_models_cli._internal.aws import validate_credentials

    validate_credentials()


def add_validate_aws_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "validate_aws_credentials",
        help="Validate and refresh AWS credentials for internal release access.",
        description="Ensure the 'qaihm' AWS profile has valid credentials. "
        "If credentials are expired, refreshes them via saml2aws. "
        "Requires the [internal] extra (pip install qai_hub_models_cli[internal]).",
    )
    parser.set_defaults(func=_run_validate_aws)
    return parser


def main(args: list[str] | None = None) -> None:
    _check_version_match()

    parser = argparse.ArgumentParser(
        prog="qai_hub_models",
        description="Qualcomm AI Hub Models CLI.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {CURRENT_VERSION}",
    )
    subparsers = parser.add_subparsers()

    add_fetch_parser(subparsers)
    add_info_parser(subparsers)
    add_perf_parser(subparsers)
    add_numerics_parser(subparsers)
    add_list_models_parser(subparsers)
    add_list_devices_parser(subparsers)
    add_list_chipsets_parser(subparsers)
    add_list_runtimes_parser(subparsers)
    add_versions_parser(subparsers)
    if use_internal_releases() or is_internal_repo():
        add_validate_aws_parser(subparsers)

    parsed = parser.parse_args(args)
    if hasattr(parsed, "func"):
        try:
            parsed.func(parsed)
        except Exception as e:
            if bool_envvar_value(VERBOSE_EXCEPTIONS_ENVVAR):
                raise
            print(e)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
