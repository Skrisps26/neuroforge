"""Small, dependency-light command-line entry points for NeuroForge stages.

Heavy study integrations are deliberately imported inside the command that uses them so
``neuroforge doctor`` remains usable in the data environment.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any, Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _positive_elapsed(started: float) -> float:
    return max(time.monotonic() - started, sys.float_info.epsilon)


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def _memory_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _gpu_inventory() -> list[dict[str, str]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[dict[str, str]] = []
    for row in completed.stdout.splitlines():
        values = [value.strip() for value in row.split(",")]
        if len(values) == 3:
            rows.append({"name": values[0], "memory_total": values[1], "driver_version": values[2]})
    return rows


def _hardware() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "memory_bytes": _memory_bytes(),
        "gpus": _gpu_inventory(),
    }


def _tool_presence() -> dict[str, bool]:
    return {name: shutil.which(name) is not None for name in ("git", "nvidia-smi", "podman", "docker")}


def _auth_presence() -> dict[str, bool]:
    # Deliberately inspect environment-variable presence only: neither values nor
    # credential files are read by a non-billable diagnostic command.
    names = (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_PROFILE",
        "RUNPOD_API_KEY",
    )
    return {name: bool(os.environ.get(name)) for name in names}


def _libraries() -> dict[str, str]:
    names = (
        "neuroforge",
        "pydantic",
        "typer",
        "PyYAML",
        "torch",
        "transformers",
        "trl",
        "peft",
        "bitsandbytes",
        "accelerate",
        "auditkit",
        "safetune",
        "circuitkit",
        "curatorkit",
        "datasets",
        "huggingface_hub",
        "scipy",
        "scikit-learn",
    )
    return {name: _version(name) for name in names}


def _file_id(path: Path) -> str:
    from .artifacts import sha256_file

    return f"{path}:{sha256_file(path)}"


def _artifact_id(path: Path) -> str:
    """Return a real hash identifier for a file or immutable adapter directory."""
    if path.is_file():
        return _file_id(path)
    if not path.is_dir():
        raise ValueError(f"artifact path is neither a file nor a directory: {path}")
    from .artifacts import sha256_file
    from .contracts import canonical_json

    files = [candidate for candidate in sorted(path.rglob("*")) if candidate.is_file()]
    if not files:
        raise ValueError(f"artifact directory contains no files: {path}")
    entries = [
        {"path": str(candidate.relative_to(path)), "sha256": sha256_file(candidate)}
        for candidate in files
    ]
    return f"{path}:{sha256(canonical_json(entries).encode('utf-8')).hexdigest()}"


def _error_config_hash(config_path: Path | None, command: str) -> str:
    if config_path is not None:
        try:
            return sha256(config_path.read_bytes()).hexdigest()
        except OSError:
            pass
    return sha256(f"neuroforge:{command}".encode("utf-8")).hexdigest()


def _load_config(config_path: Path):
    from .contracts import load_config

    return load_config(config_path)


def _configured_fault(config: Any, fault_id: str) -> Any:
    matches = [fault for fault in config.faults if fault.fault_id == fault_id]
    if len(matches) != 1:
        raise ValueError(f"fault_id must identify exactly one configured fault: {fault_id}")
    return matches[0]


def _load_curated_split(data_manifest: Path, split: str, expected_config_hash: str):
    """Load only a hash-verified nonsealed split from a curate manifest."""
    if split not in {"train", "development", "validation"}:
        raise ValueError("sealed and extension data are restricted to the independent evaluator")
    from .artifacts import sha256_file
    from .data import load_items

    manifest = _read_json(data_manifest)
    if not isinstance(manifest, dict) or manifest.get("config_hash") != expected_config_hash:
        raise ValueError("data manifest is missing or does not match the supplied study configuration")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not isinstance(splits.get(split), dict):
        raise ValueError(f"data manifest has no {split} split")
    details = splits[split]
    expected_hash = details.get("sha256")
    expected_ids = details.get("sample_ids")
    split_path = data_manifest.parent / f"{split}.jsonl"
    if not isinstance(expected_hash, str) or sha256_file(split_path) != expected_hash:
        raise ValueError(f"{split} split does not match its immutable manifest hash")
    items = load_items(split_path)
    if expected_ids != [item.sample_id for item in items]:
        raise ValueError(f"{split} split sample IDs do not match its immutable manifest")
    return items, split_path


def _run_snapshot(config: Any, run_dir: Path) -> tuple[str, dict[str, str]]:
    """Create or verify the immutable static run provenance snapshot."""
    from .artifacts import read_json_artifact, write_json_artifact
    from .contracts import canonical_json, config_hash

    config_payload = config.model_dump(mode="json")
    digest = config_hash(config_payload)
    libraries = _libraries()
    payload = {
        "schema_version": 1,
        "config_hash": digest,
        "git_commit": _git_commit(),
        "model": {"id": config.model_id, "revision": config.model_revision},
        "guardian": {"id": config.guardian_id, "revision": config.guardian_revision},
        "datasets": {
            "xstest": {"id": config.xstest_dataset_id, "revision": config.xstest_revision},
            "gsm8k": {"id": config.gsm8k_dataset_id, "revision": config.gsm8k_revision},
        },
        "library_versions": libraries,
        "hardware": _hardware(),
        "quantization": config.quantization,
        "generation": {
            "seed": config.seed,
            "strategy": config.generation_strategy,
            "batch_size": config.generation_batch_size,
            "do_sample": config.generation_do_sample,
            "temperature": config.generation_temperature,
            "top_p": config.generation_top_p,
        },
        "resource_outputs": [],
        "config": config_payload,
    }
    destination = run_dir / "provenance" / "run-config.json"
    if destination.exists():
        existing = read_json_artifact(destination)
        if (
            existing.get("config_hash") != digest
            or canonical_json(existing.get("config")) != canonical_json(config_payload)
        ):
            raise ValueError(f"run directory has a different immutable configuration snapshot: {destination}")
    else:
        write_json_artifact(destination, payload, root=run_dir)
    return digest, libraries


def _stage_snapshot(
    config: Any, run_dir: Path, command: str, resource_outputs: list[str]
) -> str:
    from .artifacts import write_json_artifact

    destination = run_dir / "provenance" / f"{command}.json"
    payload = {
        "schema_version": 1,
        "config_hash": _run_snapshot(config, run_dir)[0],
        "git_commit": _git_commit(),
        "model": {"id": config.model_id, "revision": config.model_revision},
        "datasets": {
            "xstest_revision": config.xstest_revision,
            "gsm8k_revision": config.gsm8k_revision,
        },
        "library_versions": _libraries(),
        "hardware": _hardware(),
        "quantization": config.quantization,
        "generation": {
            "seed": config.seed,
            "strategy": config.generation_strategy,
            "batch_size": config.generation_batch_size,
            "do_sample": config.generation_do_sample,
            "temperature": config.generation_temperature,
            "top_p": config.generation_top_p,
        },
        "resource_outputs": resource_outputs,
    }
    return str(destination) if destination.exists() else _write_json(destination, payload, run_dir)


def _write_json(path: Path, value: Any, run_dir: Path) -> str:
    from .artifacts import write_json_artifact

    write_json_artifact(path, value, root=run_dir)
    return str(path)


def _persist_result(run_dir: Path, command: str, result: Any, *, failure: bool = False) -> str:
    destination = run_dir / "stages" / (
        f"{command}-error-{uuid.uuid4().hex}.json" if failure else f"{command}.json"
    )
    return _write_json(destination, result.model_dump(mode="json"), run_dir)


def _emit_and_exit(run_dir: Path, command: str, result: Any) -> None:
    try:
        record_path = _persist_result(run_dir, command, result, failure=result.status == "error")
    except Exception as error:
        typer.echo(f"{command}: unable to persist stage result: {type(error).__name__}: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps({"stage_result": record_path, **result.model_dump(mode="json")}, sort_keys=True))
    if result.status != "success":
        raise typer.Exit(1)


def _error_result(command: str, config_path: Path | None, started: float, error: Exception):
    from .contracts import StageResult

    return StageResult(
        status="error",
        input_artifact_ids=[str(config_path)] if config_path is not None else [],
        output_artifact_ids=[],
        config_hash=_error_config_hash(config_path, command),
        library_versions=_libraries(),
        timing={"wall_seconds": _positive_elapsed(started)},
        errors=[f"{type(error).__name__}: {error}"],
    )


def _command_failure(command: str, config_path: Path | None, run_dir: Path, started: float, error: Exception) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _emit_and_exit(run_dir, command, _error_result(command, config_path, started, error))


@app.command()
def doctor(
    run_dir: Annotated[Path, typer.Option("--run-dir", file_okay=False, dir_okay=True)],
) -> None:
    """Record local hardware, tools, and credential-presence booleans without network calls."""
    started = time.monotonic()
    try:
        from .contracts import StageResult, config_hash

        run_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "hardware": _hardware(),
            "tools": _tool_presence(),
            "auth_configured": _auth_presence(),
            "library_versions": _libraries(),
            "network_calls": False,
        }
        report_path = _write_json(run_dir / "doctor" / "report.json", report, run_dir)
        result = StageResult(
            status="success",
            input_artifact_ids=[],
            output_artifact_ids=[report_path],
            config_hash=config_hash({"command": "doctor"}),
            library_versions=_libraries(),
            timing={"wall_seconds": _positive_elapsed(started)},
        )
        _emit_and_exit(run_dir, "doctor", result)
    except typer.Exit:
        raise
    except Exception as error:
        _command_failure("doctor", None, run_dir, started, error)


@app.command()
def curate(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True, dir_okay=False)],
    run_dir: Annotated[Path, typer.Option("--run-dir", file_okay=False, dir_okay=True)],
) -> None:
    """Curate pinned source data into immutable leakage-safe splits."""
    started = time.monotonic()
    try:
        study = _load_config(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        _run_snapshot(study, run_dir)
        from .data import curate as curate_data

        result = curate_data(study, run_dir)
        provenance = _stage_snapshot(study, run_dir, "curate", result.output_artifact_ids)
        result = result.model_copy(update={"output_artifact_ids": [*result.output_artifact_ids, provenance]})
        _emit_and_exit(run_dir, "curate", result)
    except typer.Exit:
        raise
    except Exception as error:
        _command_failure("curate", config, run_dir, started, error)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _response_records(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, dict):
        payload = payload.get("responses")
    if not isinstance(payload, list) or any(not isinstance(record, dict) for record in payload):
        raise ValueError("responses must be a JSON array or an artifact object containing a responses array")
    return payload


@app.command("route")
def route(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True, dir_okay=False)],
    run_dir: Annotated[Path, typer.Option("--run-dir", file_okay=False, dir_okay=True)],
    features: Annotated[Path, typer.Option("--features", exists=True, readable=True, dir_okay=False)],
    candidates: Annotated[Path, typer.Option("--candidates", exists=True, readable=True, dir_okay=False)],
    drift_metrics: Annotated[Path, typer.Option("--drift-metrics", exists=True, readable=True, dir_okay=False)],
    observed_faults: Annotated[Path, typer.Option("--observed-faults", exists=True, readable=True, dir_okay=False)],
    base_metrics: Annotated[Path | None, typer.Option("--base-metrics", exists=True, readable=True, dir_okay=False)] = None,
    diagnostic_costs: Annotated[Path | None, typer.Option("--diagnostic-costs", exists=True, readable=True, dir_okay=False)] = None,
) -> None:
    """Analyze only supplied measured diagnostics, candidates, metrics, and observed faults."""
    started = time.monotonic()
    inputs = [features, candidates, drift_metrics, observed_faults]
    if base_metrics is not None:
        inputs.append(base_metrics)
    if diagnostic_costs is not None:
        inputs.append(diagnostic_costs)
    try:
        study = _load_config(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        digest, libraries = _run_snapshot(study, run_dir)
        from .contracts import CandidateRecord, DiagnosticFeatures, Metrics, StageResult
        from .router import analyze_router

        loaded_features = [DiagnosticFeatures.model_validate(value) for value in _read_json(features)]
        loaded_candidates = [CandidateRecord.model_validate(value) for value in _read_json(candidates)]
        loaded_drift = {
            fault_id: Metrics.model_validate(value) for fault_id, value in _read_json(drift_metrics).items()
        }
        loaded_observed = _read_json(observed_faults)
        if not isinstance(loaded_observed, list) or not all(isinstance(value, str) for value in loaded_observed):
            raise ValueError("observed-faults must be a JSON array of explicit fault IDs")
        loaded_base = (
            {fault_id: Metrics.model_validate(value) for fault_id, value in _read_json(base_metrics).items()}
            if base_metrics is not None
            else None
        )
        loaded_costs = _read_json(diagnostic_costs) if diagnostic_costs is not None else None
        analysis = analyze_router(
            loaded_features,
            loaded_candidates,
            loaded_drift,
            observed_fault_ids=loaded_observed,
            base_metrics=loaded_base,
            diagnostic_cost_usd=loaded_costs,
            seed=study.seed,
        )
        analysis_path = _write_json(run_dir / "route" / "analysis.json", analysis, run_dir)
        provenance = _stage_snapshot(study, run_dir, "route", [analysis_path])
        status = "success" if analysis["status"] == "success" else "inconclusive"
        result = StageResult(
            status=status,
            input_artifact_ids=[_file_id(config), *[_file_id(path) for path in inputs]],
            output_artifact_ids=[analysis_path, provenance],
            config_hash=digest,
            library_versions=libraries,
            timing={"wall_seconds": _positive_elapsed(started)},
            warnings=analysis.get("reasons", []),
        )
        _emit_and_exit(run_dir, "route", result)
    except typer.Exit:
        raise
    except Exception as error:
        _command_failure("route", config, run_dir, started, error)


@app.command()
def audit(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True, dir_okay=False)],
    run_dir: Annotated[Path, typer.Option("--run-dir", file_okay=False, dir_okay=True)],
    data_manifest: Annotated[
        Path, typer.Option("--data-manifest", exists=True, readable=True, dir_okay=False)
    ],
    responses: Annotated[Path, typer.Option("--responses", exists=True, readable=True, dir_okay=False)],
    split: Annotated[str, typer.Option("--split")],
    guardian_model: Annotated[
        Path | None, typer.Option("--guardian-model", exists=True, readable=True, file_okay=False)
    ] = None,
) -> None:
    """Score frozen development/validation responses; sealed scoring requires the independent evaluator."""
    started = time.monotonic()
    try:
        if split == "sealed":
            raise PermissionError(
                "sealed audit is fail-closed: use the independent evaluator boundary, not this CLI"
            )
        if split not in {"development", "validation"}:
            raise ValueError("split must be development, validation, or sealed")
        study = _load_config(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        digest, libraries = _run_snapshot(study, run_dir)
        from .contracts import StageResult
        from .evaluation import evaluate_responses

        loaded_items, split_path = _load_curated_split(data_manifest, split, digest)
        loaded_responses = _response_records(responses)
        metrics = evaluate_responses(
            loaded_items,
            loaded_responses,
            guardian_model=str(guardian_model) if guardian_model is not None else None,
            guardian_revision=study.guardian_revision,
            guardian_device="cuda" if study.mode == "cloud" else "cpu",
        )
        metrics_path = _write_json(
            run_dir / "audit" / f"{split}-metrics.json",
            {"split": split, "metrics": metrics.model_dump(mode="json")},
            run_dir,
        )
        provenance = _stage_snapshot(study, run_dir, "audit", [metrics_path])
        status = "success" if metrics.complete else "inconclusive"
        result = StageResult(
            status=status,
            input_artifact_ids=[
                _file_id(config),
                _file_id(data_manifest),
                _file_id(split_path),
                _file_id(responses),
            ],
            output_artifact_ids=[metrics_path, provenance],
            config_hash=digest,
            library_versions=libraries,
            timing={"wall_seconds": _positive_elapsed(started)},
            warnings=[
                "audit scores precomputed response records only; it does not generate model responses",
                *metrics.evaluation_errors,
            ],
        )
        _emit_and_exit(run_dir, "audit", result)
    except typer.Exit:
        raise
    except Exception as error:
        _command_failure("audit", config, run_dir, started, error)


@app.command()
def smoke(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True, dir_okay=False)],
    run_dir: Annotated[Path, typer.Option("--run-dir", file_okay=False, dir_okay=True)],
) -> None:
    """Run the bounded model feasibility generation and persist its observed outcome."""
    started = time.monotonic()
    try:
        study = _load_config(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        _run_snapshot(study, run_dir)
        from .modeling import smoke as model_smoke

        result = model_smoke(study, run_dir)
        provenance = _stage_snapshot(study, run_dir, "smoke", result.output_artifact_ids)
        result = result.model_copy(update={"output_artifact_ids": [*result.output_artifact_ids, provenance]})
        _emit_and_exit(run_dir, "smoke", result)
    except typer.Exit:
        raise
    except Exception as error:
        _command_failure("smoke", config, run_dir, started, error)


@app.command()
def baseline(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True, dir_okay=False)],
    run_dir: Annotated[Path, typer.Option("--run-dir", file_okay=False, dir_okay=True)],
    data_manifest: Annotated[
        Path, typer.Option("--data-manifest", exists=True, readable=True, dir_okay=False)
    ],
    guardian_model: Annotated[
        Path, typer.Option("--guardian-model", exists=True, readable=True, file_okay=False)
    ],
) -> None:
    """Generate and AuditKIT-score only manifest-verified development and validation splits."""
    started = time.monotonic()
    try:
        study = _load_config(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        digest, libraries = _run_snapshot(study, run_dir)
        from .contracts import StageResult
        from .evaluation import evaluate_responses
        from .modeling import generate_responses, load_model

        development, development_path = _load_curated_split(data_manifest, "development", digest)
        validation, validation_path = _load_curated_split(data_manifest, "validation", digest)
        splits = {"development": development, "validation": validation}
        model, tokenizer = load_model(study)
        outputs: list[str] = []
        metrics_by_split: dict[str, Any] = {}
        evaluation_errors: list[str] = []
        for split, items_for_split in splits.items():
            metadata: dict[str, Any] = {}
            responses = generate_responses(
                model, tokenizer, items_for_split, study, metadata_out=metadata
            )
            response_path = _write_json(
                run_dir / "baseline" / f"{split}-responses.json",
                {"split": split, "responses": responses},
                run_dir,
            )
            metadata_path = _write_json(
                run_dir / "baseline" / f"{split}-generation-metadata.json",
                {"split": split, "metadata": metadata},
                run_dir,
            )
            metrics = evaluate_responses(
                items_for_split,
                responses,
                guardian_model=str(guardian_model),
                guardian_revision=study.guardian_revision,
                guardian_device="cuda" if study.mode == "cloud" else "cpu",
            )
            metrics_path = _write_json(
                run_dir / "baseline" / f"{split}-metrics.json",
                {"split": split, "metrics": metrics.model_dump(mode="json")},
                run_dir,
            )
            outputs.extend((response_path, metadata_path, metrics_path))
            metrics_by_split[split] = metrics
            evaluation_errors.extend(f"{split}:{error}" for error in metrics.evaluation_errors)
        provenance = _stage_snapshot(study, run_dir, "baseline", outputs)
        status = "success" if all(metrics.complete for metrics in metrics_by_split.values()) else "inconclusive"
        result = StageResult(
            status=status,
            input_artifact_ids=[
                _file_id(config),
                _file_id(data_manifest),
                _file_id(development_path),
                _file_id(validation_path),
            ],
            output_artifact_ids=[*outputs, provenance],
            config_hash=digest,
            library_versions=libraries,
            timing={"wall_seconds": _positive_elapsed(started)},
            warnings=evaluation_errors,
        )
        _emit_and_exit(run_dir, "baseline", result)
    except typer.Exit:
        raise
    except Exception as error:
        _command_failure("baseline", config, run_dir, started, error)


@app.command("make-faults")
def make_faults(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True, dir_okay=False)],
    run_dir: Annotated[Path, typer.Option("--run-dir", file_okay=False, dir_okay=True)],
    data_manifest: Annotated[
        Path, typer.Option("--data-manifest", exists=True, readable=True, dir_okay=False)
    ],
    fault_id: Annotated[str, typer.Option("--fault-id")],
) -> None:
    """Train one configured LoRA fault from the manifest-verified train split only."""
    started = time.monotonic()
    stage = f"make-faults-{fault_id}"
    try:
        study = _load_config(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        digest, _ = _run_snapshot(study, run_dir)
        fault = _configured_fault(study, fault_id)
        train_items, train_path = _load_curated_split(data_manifest, "train", digest)
        from .training import train_fault

        result = train_fault(study, fault, train_items, run_dir)
        provenance = _stage_snapshot(study, run_dir, stage, result.output_artifact_ids)
        result = result.model_copy(
            update={
                "input_artifact_ids": [
                    _file_id(config),
                    _file_id(data_manifest),
                    _file_id(train_path),
                ],
                "output_artifact_ids": [*result.output_artifact_ids, provenance],
            }
        )
        _emit_and_exit(run_dir, stage, result)
    except typer.Exit:
        raise
    except Exception as error:
        _command_failure(stage, config, run_dir, started, error)


@app.command()
def repair(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True, dir_okay=False)],
    run_dir: Annotated[Path, typer.Option("--run-dir", file_okay=False, dir_okay=True)],
    data_manifest: Annotated[
        Path, typer.Option("--data-manifest", exists=True, readable=True, dir_okay=False)
    ],
    fault_id: Annotated[str, typer.Option("--fault-id")],
    method: Annotated[str, typer.Option("--method")],
    adapter_path: Annotated[
        Path, typer.Option("--adapter-path", exists=True, readable=True, file_okay=False)
    ],
    diagnostics: Annotated[
        Path, typer.Option("--diagnostics", exists=True, readable=True, dir_okay=False)
    ],
) -> None:
    """Build one registered repair from immutable fault and diagnostic artifacts."""
    started = time.monotonic()
    stage = f"repair-{fault_id}-{method}"
    try:
        if method not in {"B0", "M1", "M2", "M3", "R0", "R1", "R2"}:
            raise ValueError(f"unregistered repair method: {method}")
        study = _load_config(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        digest, _ = _run_snapshot(study, run_dir)
        fault = _configured_fault(study, fault_id)
        development_items, development_path = _load_curated_split(
            data_manifest, "development", digest
        )
        diagnostic_values = _read_json(diagnostics)
        if not isinstance(diagnostic_values, dict):
            raise ValueError("diagnostics must be a JSON object from a measured diagnostic artifact")
        from .repairs import repair_candidate

        result = repair_candidate(
            study,
            fault,
            method,
            adapter_path,
            run_dir,
            diagnostics=diagnostic_values,
            development_items=development_items,
        )
        provenance = _stage_snapshot(study, run_dir, stage, result.output_artifact_ids)
        result = result.model_copy(
            update={
                "input_artifact_ids": [
                    _file_id(config),
                    _file_id(data_manifest),
                    _file_id(development_path),
                    _artifact_id(adapter_path),
                    _file_id(diagnostics),
                ],
                "output_artifact_ids": [*result.output_artifact_ids, provenance],
            }
        )
        _emit_and_exit(run_dir, stage, result)
    except typer.Exit:
        raise
    except Exception as error:
        _command_failure(stage, config, run_dir, started, error)
