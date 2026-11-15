#!/usr/bin/env python3
"""Deterministic, non-sealed NeuroForge study execution driver."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from neuroforge.artifacts import read_json_artifact, read_manifest, sha256_file
from neuroforge.cli import (
    _artifact_id,
    _configured_fault,
    _file_id,
    _load_config,
    _load_curated_split,
    _run_snapshot,
    _stage_snapshot,
    _write_json,
)
from neuroforge.contracts import CandidateRecord, GateDecision, GateOutcome, Metrics, StageResult

_EXPORT_RESERVE_SECONDS = 60.0
_METHODS = ("B0", "M1", "M2", "M3", "R0", "R1", "R2")
_PER_FAULT_PHASES = frozenset({"population", "diagnostics", "candidates", "sealed"})


def _stage_key(args: argparse.Namespace) -> str | None:
    """Fault-scoped stage identity. Only per-fault phases isolate records by fault."""
    if args.phase in _PER_FAULT_PHASES:
        return args.fault_id
    return None


def _parse_deadline(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("--deadline must be a UTC ISO-8601 timestamp")
    return parsed.astimezone(UTC)


def _parse_budget(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        budget = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("--hard-budget must be a non-negative decimal") from error
    if not budget.is_finite() or budget < 0:
        raise ValueError("--hard-budget must be a non-negative decimal")
    return budget


def _release() -> None:
    """Collect memory after callers have dropped all model references."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _guardian_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "granite-guardian"


def _evaluate_and_capture(
    config: Any, items: list[Any], responses: list[dict[str, Any]]
) -> tuple[Metrics, list[dict[str, Any]]]:
    """Score with AuditKIT and retain only scorer-emitted per-record values."""
    from neuroforge.evaluation import evaluate_responses

    score_records: list[dict[str, Any]] = []
    metrics = evaluate_responses(
        items,
        responses,
        guardian_model=str(_guardian_path()),
        guardian_revision=config.guardian_revision,
        guardian_device="cuda" if config.mode == "cloud" else "cpu",
        score_records_out=score_records,
    )
    return metrics, score_records


def _stage_path(run_dir: Path, phase: str, fault_id: str | None = None) -> Path:
    suffix = f"-{fault_id}" if fault_id is not None else ""
    return run_dir / "study-execution" / f"{phase}{suffix}.json"


def _record_stage(
    run_dir: Path,
    phase: str,
    result: StageResult,
    *,
    fault_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> StageResult:
    payload = {
        "phase": phase,
        "fault_id": fault_id,
        "result": result.model_dump(mode="json"),
        "details": details or {},
    }
    _write_json(_stage_path(run_dir, phase, fault_id), payload, run_dir)
    return result


def _existing_stage(
    run_dir: Path, phase: str, config_hash: str, fault_id: str | None = None
) -> StageResult | None:
    path = _stage_path(run_dir, phase, fault_id)
    if not path.exists():
        return None
    payload = read_json_artifact(path, root=run_dir)
    result = StageResult.model_validate(payload.get("result"))
    if result.config_hash != config_hash:
        raise ValueError(f"existing {phase} stage has a different configuration hash: {path}")
    return result


def _result(
    *,
    status: str,
    config_hash: str,
    libraries: dict[str, str],
    started: float,
    inputs: Sequence[str] = (),
    outputs: Sequence[str] = (),
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
) -> StageResult:
    return StageResult(
        status=status,
        input_artifact_ids=list(inputs),
        output_artifact_ids=list(outputs),
        config_hash=config_hash,
        library_versions=libraries,
        timing={"wall_seconds": max(time.monotonic() - started, sys.float_info.epsilon)},
        warnings=list(warnings),
        errors=list(errors),
    )


def _can_start(deadline: datetime | None) -> str | None:
    if deadline is None:
        return None
    remaining = (deadline - datetime.now(UTC)).total_seconds()
    if remaining <= _EXPORT_RESERVE_SECONDS:
        return f"deadline guard prevented a new bounded stage; remaining_seconds={remaining:.3f}"
    return None


def _recorded_candidate_cost(run_dir: Path) -> Decimal:
    """Sum only durable candidate records; this runner never queries a provider."""
    total = Decimal("0")
    for path in (run_dir / "repairs").glob("*/*/validation-record.json"):
        payload = read_json_artifact(path, root=run_dir)
        total += CandidateRecord.model_validate(payload["candidate"]).cost_usd
    return total


def _deadline_result(
    config_hash: str, libraries: dict[str, str], started: float, reason: str
) -> StageResult:
    return _result(
        status="inconclusive",
        config_hash=config_hash,
        libraries=libraries,
        started=started,
        warnings=[reason],
    )


def _baseline_manifest_path(run_dir: Path) -> Path:
    return run_dir / "baseline" / "manifest.json"


def _baseline(
    run_dir: Path,
    config: Any,
    data_manifest: Path,
    digest: str,
    libraries: dict[str, str],
    deadline: datetime | None,
) -> tuple[StageResult, dict[str, Metrics]]:
    existing = _existing_stage(run_dir, "baseline", digest)
    manifest_path = _baseline_manifest_path(run_dir)
    if existing is not None:
        if not manifest_path.exists():
            raise ValueError("completed baseline stage is missing its immutable baseline manifest")
        return existing, _read_baseline(manifest_path, digest)
    if manifest_path.exists():
        # A manifest is sufficient to resume only when it proves the exact frozen inputs.
        metrics = _read_baseline(manifest_path, digest)
        started = time.monotonic()
        result = _result(
            status="success"
            if all(value.complete for value in metrics.values())
            else "inconclusive",
            config_hash=digest,
            libraries=libraries,
            started=started,
            inputs=[_file_id(data_manifest)],
            outputs=[str(manifest_path)],
        )
        return _record_stage(run_dir, "baseline", result), metrics
    reason = _can_start(deadline)
    started = time.monotonic()
    if reason is not None:
        return _record_stage(
            run_dir, "baseline", _deadline_result(digest, libraries, started, reason)
        ), {}

    from neuroforge.modeling import generate_responses, load_model

    development, development_path = _load_curated_split(data_manifest, "development", digest)
    validation, validation_path = _load_curated_split(data_manifest, "validation", digest)
    model, tokenizer = load_model(config)
    generated: dict[str, tuple[list[Any], list[dict[str, Any]], dict[str, Any]]] = {}
    try:
        for split, items in (("development", development), ("validation", validation)):
            metadata: dict[str, Any] = {}
            generated[split] = (
                items,
                generate_responses(model, tokenizer, items, config, metadata_out=metadata),
                metadata,
            )
    finally:
        del model, tokenizer
        _release()

    outputs: list[str] = []
    metrics: dict[str, Metrics] = {}
    hashes: dict[str, str] = {}
    for split, (items, responses, metadata) in generated.items():
        response_path = run_dir / "baseline" / f"{split}-responses.json"
        metadata_path = run_dir / "baseline" / f"{split}-generation-metadata.json"
        score_path = run_dir / "baseline" / f"{split}-score-records.json"
        metric_path = run_dir / "baseline" / f"{split}-metrics.json"
        _write_json(response_path, {"split": split, "responses": responses}, run_dir)
        _write_json(metadata_path, {"split": split, "metadata": metadata}, run_dir)
        measured, score_records = _evaluate_and_capture(config, items, responses)
        _write_json(score_path, {"split": split, "score_records": score_records}, run_dir)
        _write_json(
            metric_path, {"split": split, "metrics": measured.model_dump(mode="json")}, run_dir
        )
        metrics[split] = measured
        for path in (response_path, metadata_path, score_path, metric_path):
            hashes[str(path.relative_to(run_dir))] = sha256_file(path)
            outputs.append(str(path))
    payload = {
        "schema_version": 1,
        "config_hash": digest,
        "data_manifest": _file_id(data_manifest),
        "split_artifacts": {
            "development": _file_id(development_path),
            "validation": _file_id(validation_path),
        },
        "metrics": {name: value.model_dump(mode="json") for name, value in metrics.items()},
        "artifact_sha256": hashes,
    }
    _write_json(manifest_path, payload, run_dir)
    outputs.append(str(manifest_path))
    provenance = _stage_snapshot(config, run_dir, "baseline", outputs)
    outputs.append(provenance)
    result = _result(
        status="success" if all(value.complete for value in metrics.values()) else "inconclusive",
        config_hash=digest,
        libraries=libraries,
        started=started,
        inputs=[_file_id(data_manifest), _file_id(development_path), _file_id(validation_path)],
        outputs=outputs,
        warnings=[error for value in metrics.values() for error in value.evaluation_errors],
    )
    return _record_stage(run_dir, "baseline", result), metrics


def _read_baseline(path: Path, digest: str) -> dict[str, Metrics]:
    payload = read_json_artifact(path)
    if payload.get("config_hash") != digest:
        raise ValueError("baseline artifacts have a different configuration hash")
    hashes = payload.get("artifact_sha256")
    metrics = payload.get("metrics")
    if not isinstance(hashes, dict) or not isinstance(metrics, dict):
        raise ValueError("baseline manifest is incomplete")
    root = path.parents[1]
    for relative, expected in hashes.items():
        artifact = root / relative
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or sha256_file(artifact) != expected
        ):
            raise ValueError(f"baseline artifact hash mismatch: {relative}")
    required = {"development", "validation"}
    if set(metrics) != required:
        raise ValueError(
            "baseline manifest does not contain both development and validation metrics"
        )
    return {name: Metrics.model_validate(metrics[name]) for name in sorted(required)}


def _fault_validation_path(run_dir: Path, fault_id: str) -> Path:
    return run_dir / "faults" / fault_id / "validation.json"


def _load_fault_validation(run_dir: Path, fault_id: str, digest: str) -> dict[str, Any] | None:
    path = _fault_validation_path(run_dir, fault_id)
    if not path.exists():
        return None
    payload = read_json_artifact(path, root=run_dir)
    if payload.get("config_hash") != digest:
        raise ValueError(f"fault validation has a different configuration hash: {path}")
    return payload


def _observed_eligibility(fault: Any, base: Metrics, drift: Metrics) -> tuple[bool, str]:
    if not base.complete or not drift.complete:
        return False, "incomplete base or drift validation metrics"
    if fault.fault_type == "F1":
        if (
            drift.safety_failure_rate is not None
            and base.safety_failure_rate is not None
            and drift.safety_failure_rate > base.safety_failure_rate
        ):
            return True, "observed strict safety failure increase"
        return False, "null fault: no observed strict safety failure increase"
    if fault.fault_type == "F2":
        if (
            drift.overrefusal_rate is not None
            and base.overrefusal_rate is not None
            and drift.overrefusal_rate > base.overrefusal_rate
        ):
            return True, "observed strict overrefusal increase"
        return False, "null fault: no observed strict overrefusal increase"
    return False, f"unsupported fault type: {fault.fault_type}"


def _train_and_measure_fault(
    run_dir: Path,
    config: Any,
    fault: Any,
    data_manifest: Path,
    digest: str,
    libraries: dict[str, str],
    baseline: Metrics,
    deadline: datetime | None,
) -> tuple[dict[str, Any] | None, StageResult | None]:
    prior = _load_fault_validation(run_dir, fault.fault_id, digest)
    if prior is not None:
        return prior, None
    reason = _can_start(deadline)
    started = time.monotonic()
    if reason is not None:
        return None, _deadline_result(digest, libraries, started, reason)
    from neuroforge.modeling import generate_responses, load_model
    from neuroforge.training import train_fault

    train, train_path = _load_curated_split(data_manifest, "train", digest)
    trained = train_fault(config, fault, train, run_dir)
    if trained.status != "success":
        return None, trained
    adapter_path = run_dir / "faults" / fault.fault_id / "adapter"
    validation, validation_path = _load_curated_split(data_manifest, "validation", digest)
    model, tokenizer = load_model(config, adapter_path=adapter_path)
    try:
        metadata: dict[str, Any] = {}
        responses = generate_responses(model, tokenizer, validation, config, metadata_out=metadata)
    finally:
        del model, tokenizer
        _release()
    drift, score_records = _evaluate_and_capture(config, validation, responses)
    eligible, eligibility_reason = _observed_eligibility(fault, baseline, drift)
    response_path = run_dir / "faults" / fault.fault_id / "validation-responses.json"
    metadata_path = run_dir / "faults" / fault.fault_id / "validation-generation-metadata.json"
    score_path = run_dir / "faults" / fault.fault_id / "validation-score-records.json"
    _write_json(response_path, {"split": "validation", "responses": responses}, run_dir)
    _write_json(metadata_path, {"split": "validation", "metadata": metadata}, run_dir)
    _write_json(score_path, {"split": "validation", "score_records": score_records}, run_dir)
    payload = {
        "schema_version": 1,
        "config_hash": digest,
        "fault": fault.model_dump(mode="json"),
        "train_stage": trained.model_dump(mode="json"),
        "train_split": _file_id(train_path),
        "validation_split": _file_id(validation_path),
        "adapter": _artifact_id(adapter_path),
        "base_metrics": baseline.model_dump(mode="json"),
        "drift_metrics": drift.model_dump(mode="json"),
        "observed_eligible": eligible,
        "eligibility_reason": eligibility_reason,
        "validation_responses": _file_id(response_path),
        "validation_generation": _file_id(metadata_path),
        "validation_score_records": _file_id(score_path),
    }
    _write_json(_fault_validation_path(run_dir, fault.fault_id), payload, run_dir)
    return payload, None


def _diagnostic_path(run_dir: Path, fault_id: str) -> Path:
    return run_dir / "fault-work" / fault_id / "diagnostics" / "report.json"


def _diagnose_fault(
    run_dir: Path,
    config: Any,
    fault: Any,
    data_manifest: Path,
    digest: str,
    deadline: datetime | None,
) -> StageResult:
    existing = _diagnostic_path(run_dir, fault.fault_id)
    if existing.exists():
        report = read_json_artifact(existing, root=run_dir)
        if report.get("config_hash") != digest:
            raise ValueError(f"diagnostic report has a different configuration hash: {existing}")
        return StageResult(
            status="success" if "D2" in report and "D3" in report else "inconclusive",
            input_artifact_ids=[],
            output_artifact_ids=[str(existing)],
            config_hash=digest,
            library_versions=report.get("library_versions", {}),
            timing={"wall_seconds": sys.float_info.epsilon},
        )
    reason = _can_start(deadline)
    if reason is not None:
        return StageResult(
            status="inconclusive",
            input_artifact_ids=[],
            output_artifact_ids=[],
            config_hash=digest,
            library_versions={},
            timing={"wall_seconds": sys.float_info.epsilon},
            warnings=[reason],
        )
    from neuroforge.diagnostics import run_diagnostics
    from neuroforge.modeling import load_model
    from neuroforge.repairs import _merged_drift

    development, _ = _load_curated_split(data_manifest, "development", digest)
    adapter = run_dir / "faults" / fault.fault_id / "adapter"
    base, tokenizer = load_model(config)
    drifted = _merged_drift(config, adapter)
    setattr(drifted, "neuroforge_fault_id", fault.fault_id)
    try:
        return run_diagnostics(
            config, base, drifted, development, run_dir / "fault-work" / fault.fault_id
        )
    finally:
        del base, drifted, tokenizer
        _release()


def _candidate_path(run_dir: Path, fault_id: str, method: str) -> Path:
    return run_dir / "repairs" / fault_id / method / "validation-record.json"


def _candidate_error(
    fault: Any,
    method: str,
    error: str,
    *,
    wall_seconds: float = 0.0,
    status: GateOutcome = GateOutcome.ERROR,
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=f"{fault.fault_id}:{method}",
        fault_id=fault.fault_id,
        method=method,
        metrics=Metrics(),
        decision=GateDecision(decision=status, reasons=[error]),
        cost_usd=Decimal("0"),
        wall_seconds=wall_seconds,
        split="validation",
        seed=fault.seed,
        parameters={
            "execution_status": "blocked" if status == GateOutcome.INCONCLUSIVE else "error",
            "cost_status": "not measured",
        },
    )


def _candidate(
    run_dir: Path,
    config: Any,
    fault: Any,
    method: str,
    base: Metrics,
    drift: Metrics,
    diagnostics: dict[str, Any],
    data_manifest: Path,
    digest: str,
    deadline: datetime | None,
) -> CandidateRecord:
    path = _candidate_path(run_dir, fault.fault_id, method)
    if path.exists():
        payload = read_json_artifact(path, root=run_dir)
        if payload.get("config_hash") != digest:
            raise ValueError(f"candidate record has a different configuration hash: {path}")
        return CandidateRecord.model_validate(payload["candidate"])
    reason = _can_start(deadline)
    if reason is not None:
        return _candidate_error(fault, method, reason, status=GateOutcome.INCONCLUSIVE)
    started = time.monotonic()
    try:
        from neuroforge.modeling import generate_responses, load_model
        from neuroforge.repairs import _merged_drift, reload_candidate, repair_candidate

        development, _ = _load_curated_split(data_manifest, "development", digest)
        adapter = run_dir / "faults" / fault.fault_id / "adapter"
        built = repair_candidate(
            config,
            fault,
            method,
            adapter,
            run_dir,
            diagnostics=diagnostics,
            development_items=development,
        )
        if built.status != "success":
            record = _candidate_error(
                fault,
                method,
                "; ".join(built.errors) or f"repair status {built.status}",
                wall_seconds=time.monotonic() - started,
            )
        else:
            manifest_path = run_dir / "repairs" / fault.fault_id / method / "manifest.json"
            manifest = read_manifest(manifest_path, root=run_dir)
            validation, _ = _load_curated_split(data_manifest, "validation", digest)
            if method == "B0":
                model, tokenizer = load_model(config, adapter_path=adapter)
            else:
                base_model, tokenizer = load_model(config)
                del base_model
                _release()
                model = _merged_drift(config, adapter)
                model = reload_candidate(model, tokenizer, manifest_path.parent, manifest)
            try:
                metadata: dict[str, Any] = {}
                responses = generate_responses(
                    model, tokenizer, validation, config, metadata_out=metadata
                )
            finally:
                del model, tokenizer
                _release()
            metrics, score_records = _evaluate_and_capture(config, validation, responses)
            response_path = manifest_path.parent / "validation-responses.json"
            metadata_path = manifest_path.parent / "validation-generation-metadata.json"
            _write_json(response_path, {"split": "validation", "responses": responses}, run_dir)
            _write_json(metadata_path, {"split": "validation", "metadata": metadata}, run_dir)
            score_path = manifest_path.parent / "validation-score-records.json"
            _write_json(
                score_path, {"split": "validation", "score_records": score_records}, run_dir
            )
            from neuroforge.gate import decide

            decision = decide(
                base,
                drift,
                metrics,
                config.acceptance,
                fault_type=fault.fault_type,
                reloadable=True,
                integrity_ok=True,
                truncated_to_empty=any(bool(row.get("truncated_to_empty")) for row in responses),
            )
            record = CandidateRecord(
                candidate_id=f"{fault.fault_id}:{method}",
                fault_id=fault.fault_id,
                method=method,
                artifact_path=str(manifest_path),
                artifact_sha256=manifest.artifact_sha256,
                parameters={
                    **manifest.parameters,
                    "source_manifest_sha256": sha256_file(manifest_path),
                    "validation_responses": _file_id(response_path),
                    "validation_score_records": _file_id(score_path),
                    "cost_status": "not measured",
                },
                metrics=metrics,
                decision=decision,
                cost_usd=Decimal("0"),
                wall_seconds=time.monotonic() - started,
                split="validation",
                seed=fault.seed,
                changed_parameter_count=manifest.changed_parameter_count,
            )
    except Exception as error:
        record = _candidate_error(
            fault,
            method,
            f"{type(error).__name__}: {error}",
            wall_seconds=time.monotonic() - started,
        )
    _write_json(path, {"config_hash": digest, "candidate": record.model_dump(mode="json")}, run_dir)
    return record


def _phase_baseline(
    args: argparse.Namespace, config: Any, digest: str, libraries: dict[str, str]
) -> StageResult:
    result, _ = _baseline(
        args.run_dir, config, args.data_manifest, digest, libraries, args.deadline
    )
    return result


def _phase_first_proof(
    args: argparse.Namespace, config: Any, digest: str, libraries: dict[str, str]
) -> StageResult:
    existing = _existing_stage(args.run_dir, "first-proof", digest)
    if existing is not None:
        return existing
    started = time.monotonic()
    baseline_result, baseline = _baseline(
        args.run_dir, config, args.data_manifest, digest, libraries, args.deadline
    )
    if baseline_result.status != "success":
        return _record_stage(
            args.run_dir,
            "first-proof",
            _result(
                status="inconclusive",
                config_hash=digest,
                libraries=libraries,
                started=started,
                errors=["baseline did not complete"],
            ),
        )
    if args.fault_id not in {None, "f1-01"}:
        raise ValueError("first-proof is fixed to fault_id f1-01")
    fault = _configured_fault(config, "f1-01")
    measured, failure = _train_and_measure_fault(
        args.run_dir,
        config,
        fault,
        args.data_manifest,
        digest,
        libraries,
        baseline["validation"],
        args.deadline,
    )
    if failure is not None or measured is None:
        return _record_stage(
            args.run_dir,
            "first-proof",
            failure
            or _result(
                status="error",
                config_hash=digest,
                libraries=libraries,
                started=started,
                errors=["fault measurement missing"],
            ),
        )
    drift = Metrics.model_validate(measured["drift_metrics"])
    records = [
        _candidate(
            args.run_dir,
            config,
            fault,
            method,
            baseline["validation"],
            drift,
            {},
            args.data_manifest,
            digest,
            args.deadline,
        )
        for method in ("B0", "M1")
    ]
    status = (
        "success"
        if all(record.metrics.complete and record.decision is not None for record in records)
        else "inconclusive"
    )
    output_ids = [_file_id(_fault_validation_path(args.run_dir, fault.fault_id))]
    output_ids.extend(
        _file_id(_candidate_path(args.run_dir, fault.fault_id, method))
        for method in ("B0", "M1")
        if _candidate_path(args.run_dir, fault.fault_id, method).exists()
    )
    return _record_stage(
        args.run_dir,
        "first-proof",
        _result(
            status=status,
            config_hash=digest,
            libraries=libraries,
            started=started,
            outputs=output_ids,
        ),
    )


def _phase_population(
    args: argparse.Namespace, config: Any, digest: str, libraries: dict[str, str]
) -> StageResult:
    existing = _existing_stage(args.run_dir, "population", digest, fault_id=_stage_key(args))
    if existing is not None:
        return existing
    started = time.monotonic()
    baseline_result, baseline = _baseline(
        args.run_dir, config, args.data_manifest, digest, libraries, args.deadline
    )
    if baseline_result.status != "success":
        return _record_stage(
            args.run_dir,
            "population",
            _result(
                status="inconclusive",
                config_hash=digest,
                libraries=libraries,
                started=started,
                errors=["baseline did not complete"],
            ),
            fault_id=_stage_key(args),
        )
    failures: list[str] = []
    outputs: list[str] = []
    faults = [
        fault for fault in config.faults if args.fault_id is None or fault.fault_id == args.fault_id
    ]
    for fault in faults:
        measured, failure = _train_and_measure_fault(
            args.run_dir,
            config,
            fault,
            args.data_manifest,
            digest,
            libraries,
            baseline["validation"],
            args.deadline,
        )
        if failure is not None:
            failures.extend(failure.errors or [f"{fault.fault_id}: {failure.status}"])
            if failure.status == "inconclusive":
                break
        elif measured is not None:
            outputs.append(_file_id(_fault_validation_path(args.run_dir, fault.fault_id)))
    status = "success" if len(outputs) == len(faults) and not failures else "inconclusive"
    return _record_stage(
        args.run_dir,
        "population",
        _result(
            status=status,
            config_hash=digest,
            libraries=libraries,
            started=started,
            outputs=outputs,
            errors=failures,
        ),
        fault_id=_stage_key(args),
    )


def _phase_diagnostics(
    args: argparse.Namespace, config: Any, digest: str, libraries: dict[str, str]
) -> StageResult:
    existing = _existing_stage(args.run_dir, "diagnostics", digest, fault_id=_stage_key(args))
    if existing is not None:
        return existing
    started = time.monotonic()
    outputs: list[str] = []
    errors: list[str] = []
    for fault in config.faults:
        if args.fault_id is not None and fault.fault_id != args.fault_id:
            continue
        if _load_fault_validation(args.run_dir, fault.fault_id, digest) is None:
            errors.append(f"{fault.fault_id}: no measured fault validation")
            continue
        result = _diagnose_fault(
            args.run_dir, config, fault, args.data_manifest, digest, args.deadline
        )
        outputs.extend(result.output_artifact_ids)
        errors.extend(result.errors)
        if result.status != "success":
            errors.append(f"{fault.fault_id}: diagnostics {result.status}")
    return _record_stage(
        args.run_dir,
        "diagnostics",
        _result(
            status="success" if outputs and not errors else "inconclusive",
            config_hash=digest,
            libraries=libraries,
            started=started,
            outputs=outputs,
            errors=errors,
        ),
        fault_id=_stage_key(args),
    )


def _phase_candidates(
    args: argparse.Namespace, config: Any, digest: str, libraries: dict[str, str]
) -> StageResult:
    existing = _existing_stage(args.run_dir, "candidates", digest, fault_id=_stage_key(args))
    if existing is not None:
        return existing
    started = time.monotonic()
    baseline_result, baseline = _baseline(
        args.run_dir, config, args.data_manifest, digest, libraries, args.deadline
    )
    if baseline_result.status != "success":
        return _record_stage(
            args.run_dir,
            "candidates",
            _result(
                status="inconclusive",
                config_hash=digest,
                libraries=libraries,
                started=started,
                errors=["baseline did not complete"],
            ),
            fault_id=_stage_key(args),
        )
    outputs: list[str] = []
    errors: list[str] = []
    for fault in config.faults:
        if args.fault_id is not None and fault.fault_id != args.fault_id:
            continue
        measured = _load_fault_validation(args.run_dir, fault.fault_id, digest)
        if measured is None:
            errors.append(f"{fault.fault_id}: no measured fault validation")
            continue
        drift = Metrics.model_validate(measured["drift_metrics"])
        diagnostics: dict[str, Any] = {}
        report_path = _diagnostic_path(args.run_dir, fault.fault_id)
        if report_path.exists():
            diagnostics = read_json_artifact(report_path, root=args.run_dir)
        eligible = bool(measured["observed_eligible"])
        for method in _METHODS:
            if not eligible:
                record = _candidate_error(
                    fault,
                    method,
                    str(measured["eligibility_reason"]),
                    status=GateOutcome.INCONCLUSIVE,
                )
                candidate_path = _candidate_path(args.run_dir, fault.fault_id, method)
                if candidate_path.exists():
                    record = CandidateRecord.model_validate(
                        read_json_artifact(candidate_path, root=args.run_dir)["candidate"]
                    )
                else:
                    _write_json(
                        candidate_path,
                        {"config_hash": digest, "candidate": record.model_dump(mode="json")},
                        args.run_dir,
                    )
            else:
                record = _candidate(
                    args.run_dir,
                    config,
                    fault,
                    method,
                    baseline["validation"],
                    drift,
                    diagnostics,
                    args.data_manifest,
                    digest,
                    args.deadline,
                )
            candidate_path = _candidate_path(args.run_dir, fault.fault_id, method)
            if candidate_path.exists():
                outputs.append(_file_id(candidate_path))
            if record.decision is None or record.decision.decision in {
                GateOutcome.ERROR,
                GateOutcome.INCONCLUSIVE,
            }:
                errors.append(
                    f"{fault.fault_id}:{method}:{record.decision.decision if record.decision else 'missing decision'}"
                )
    return _record_stage(
        args.run_dir,
        "candidates",
        _result(
            status="success" if outputs and not errors else "inconclusive",
            config_hash=digest,
            libraries=libraries,
            started=started,
            outputs=outputs,
            errors=errors,
        ),
        fault_id=_stage_key(args),
    )


def _phase_route(
    args: argparse.Namespace, config: Any, digest: str, libraries: dict[str, str]
) -> StageResult:
    existing = _existing_stage(args.run_dir, "route", digest)
    if existing is not None:
        return existing
    started = time.monotonic()
    reason = _can_start(args.deadline)
    if reason is not None:
        return _record_stage(
            args.run_dir, "route", _deadline_result(digest, libraries, started, reason)
        )
    from neuroforge.router import analyze_router

    features = []
    candidates = []
    drift_metrics: dict[str, Metrics] = {}
    base_metrics: dict[str, Metrics] = {}
    observed: list[str] = []
    for fault in config.faults:
        measured = _load_fault_validation(args.run_dir, fault.fault_id, digest)
        report_path = _diagnostic_path(args.run_dir, fault.fault_id)
        if measured is None or not report_path.exists():
            continue
        report = read_json_artifact(report_path, root=args.run_dir)
        if report.get("config_hash") != digest or "features" not in report:
            continue
        from neuroforge.contracts import DiagnosticFeatures

        features.append(DiagnosticFeatures.model_validate(report["features"]))
        drift_metrics[fault.fault_id] = Metrics.model_validate(measured["drift_metrics"])
        base_metrics[fault.fault_id] = Metrics.model_validate(measured["base_metrics"])
        if measured.get("observed_eligible"):
            observed.append(fault.fault_id)
        for method in _METHODS:
            candidate_path = _candidate_path(args.run_dir, fault.fault_id, method)
            if candidate_path.exists():
                candidates.append(
                    CandidateRecord.model_validate(
                        read_json_artifact(candidate_path, root=args.run_dir)["candidate"]
                    )
                )
    analysis = analyze_router(
        features,
        candidates,
        drift_metrics,
        observed_fault_ids=observed,
        base_metrics=base_metrics,
        seed=config.seed,
    )
    if (
        len(observed) < 6
        or len({feature.fault_type for feature in features if feature.fault_id in observed}) < 2
    ):
        analysis["router_decision"] = "H3_FALLBACK"
        analysis["router_reason"] = "insufficient observed cohort; no router claim"
    output = args.run_dir / "route" / "analysis.json"
    _write_json(output, analysis, args.run_dir)
    return _record_stage(
        args.run_dir,
        "route",
        _result(
            status="success" if analysis["status"] == "success" else "inconclusive",
            config_hash=digest,
            libraries=libraries,
            started=started,
            outputs=[_file_id(output)],
            warnings=analysis.get("reasons", []),
        ),
    )


def _sealed_split_items(data_manifest: Path, digest: str) -> tuple[list[Any], Path]:
    """Load the hash-verified sealed split. Trusted parent only: repair code must never call this."""
    from neuroforge.artifacts import sha256_file
    from neuroforge.cli import _read_json  # type: ignore[attr-defined]
    from neuroforge.data import load_items

    manifest = _read_json(data_manifest)
    if not isinstance(manifest, dict) or manifest.get("config_hash") != digest:
        raise ValueError(
            "data manifest is missing or does not match the supplied study configuration"
        )
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not isinstance(splits.get("sealed"), dict):
        raise ValueError("data manifest has no sealed split")
    details = splits["sealed"]
    split_path = data_manifest.parent / "sealed.jsonl"
    if not isinstance(details.get("sha256"), str) or sha256_file(split_path) != details["sha256"]:
        raise ValueError("sealed split does not match its immutable manifest hash")
    items = load_items(split_path)
    if details.get("sample_ids") != [item.sample_id for item in items]:
        raise ValueError("sealed split sample IDs do not match its immutable manifest")
    return items, split_path


def _sealed_request_doc(config: Any, digest: str, target: str) -> dict[str, Any]:
    """Build the exact frozen evaluator request for one sealed target."""
    from neuroforge.contracts import config_hash

    if target not in ("base", "candidate"):
        raise ValueError("sealed request target must be base or candidate")
    run_config = config.model_dump(mode="json")
    if config_hash(run_config) != digest:
        raise ValueError("study configuration does not match its snapshot digest")
    return {
        "mode": "generate",
        "target": target,
        "config_hashes": {"run_config": digest},
        "run_config": run_config,
        "items_path": "sealed/sealed.jsonl",
        "responses_path": None,
    }


def _compliant_evaluator_source() -> Path:
    """Return a root-owned, non-writable tree containing the live neuroforge package."""
    import neuroforge

    package = Path(neuroforge.__file__).resolve(strict=True)
    if package.name != "__init__.py" or package.parent.name != "neuroforge":
        raise ValueError("cannot locate the live neuroforge package directory")
    live_root = package.parent.parent
    try:
        from neuroforge.isolation import _readonly_tree  # noqa: PLC2701

        return _readonly_tree(live_root, label="evaluator source")
    except Exception:
        from neuroforge.isolation import tree_sha256

        staged = Path("/opt/neuroforge-eval-src")
        if staged.exists():
            import shutil

            shutil.rmtree(staged)
        shutil.copytree(live_root, staged, symlinks=False, ignore_dangling=True)
        for root, dirs, files in os.walk(staged):
            for name in dirs + files:
                os.chmod(Path(root) / name, 0o555 if (Path(root) / name).is_dir() else 0o444)
        os.chmod(staged, 0o555)
        from neuroforge.isolation import _readonly_tree  # noqa: PLC2701

        source = _readonly_tree(staged, label="evaluator source")
        _ = tree_sha256(source)
        return source


def _run_isolated_eval(
    *,
    job_name: str,
    target: str,
    request_doc: dict[str, Any],
    fault_adapter: Path | None,
    candidate_root: Path | None,
    freeze_proof: dict[str, Any] | None,
    base_model: Path,
    guardian_model: Path,
    thresholds_root: Path,
    sealed_items_path: Path,
    config_path: Path,
    router_selection_path: Path,
) -> dict[str, Any]:
    """Stage one UID-isolated evaluator run and return its parsed output (metrics + score records only)."""
    import tempfile

    from neuroforge.isolation import (
        IsolationBlocked,
        ensure_isolation_identities,
        prepare_evaluator,
        run_evaluator,
    )

    ensure_isolation_identities()
    from neuroforge.isolation import SEAL_ROOT

    SEAL_ROOT.mkdir(parents=True, exist_ok=True)
    os.chmod(SEAL_ROOT, 0o755)
    source = _compliant_evaluator_source()
    with tempfile.TemporaryDirectory(prefix="neuroforge-sealed-inputs-") as tmp:
        inputs_root = Path(tmp)
        request_name = "request.json"
        (inputs_root / request_name).write_text(
            json.dumps(request_doc, sort_keys=True), encoding="utf-8"
        )
        proof_path: Path | None = None
        if freeze_proof is not None:
            proof_path = inputs_root / "freeze-proof.json"
            proof_path.write_text(json.dumps(freeze_proof, sort_keys=True), encoding="utf-8")
        paths = prepare_evaluator(
            job_name=job_name,
            evaluator_source=source,
            thresholds_root=thresholds_root,
            guardian_model=guardian_model,
            base_model=base_model,
            fault_adapter=fault_adapter,
            candidate_root=candidate_root,
            inputs_root=inputs_root,
            request_name=request_name,
            sealed_inputs_root=sealed_items_path.parent,
            sealed_marker=sealed_items_path.name,
            freeze_proof=proof_path,
            config_path=config_path,
            router_selection_path=router_selection_path,
            target=target,
        )
        completed = run_evaluator(paths, use_gpu=True)
        if completed.returncode != 0:
            raise IsolationBlocked(
                f"isolated evaluator failed: returncode={completed.returncode} stderr={completed.stderr[-2000:]}"
            )
        payload = json.loads(paths.output.read_text(encoding="utf-8"))
    if payload.get("status") == "ERROR":
        raise IsolationBlocked(f"isolated evaluator reported ERROR: {payload.get('error')}")
    metrics = Metrics.model_validate(payload["metrics"])
    records = payload.get("score_records")
    if not isinstance(records, list) or not records:
        raise IsolationBlocked("isolated evaluator returned no score records")
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "category",
            "error",
            "sample_id",
            "scorer",
            "unknown",
            "value",
        }:
            raise IsolationBlocked("isolated evaluator returned malformed score records")
        if "response" in record:
            raise IsolationBlocked("score records must never carry response text")
    return {"status": payload.get("status"), "metrics": metrics, "score_records": records}


def _sealed_methods(run_dir: Path, fault_id: str) -> list[str]:
    """Fixed sealed conditions: B0 plus every method with a validation-stage record."""
    methods = []
    for method in ("B0", *_METHODS[1:]):
        if (run_dir / "repairs" / fault_id / method / "validation-record.json").exists():
            methods.append(method)
    return methods


def _phase_sealed(
    args: argparse.Namespace, config: Any, digest: str, libraries: dict[str, str]
) -> StageResult:
    """Run the independent UID-isolated sealed evaluation for one fault (requires --fault-id)."""
    from neuroforge.gate import decide

    if args.fault_id is None:
        raise ValueError("sealed evaluation requires --fault-id")
    fault = _configured_fault(config, args.fault_id)
    existing = _existing_stage(args.run_dir, "sealed", digest, fault_id=fault.fault_id)
    if existing is not None:
        return existing
    started = time.monotonic()
    reason = _can_start(args.deadline)
    if reason is not None:
        return _record_stage(
            args.run_dir,
            "sealed",
            _deadline_result(digest, libraries, started, reason),
            fault_id=fault.fault_id,
        )
    analysis_path = args.run_dir / "route" / "analysis.json"
    if not analysis_path.exists():
        raise ValueError("sealed evaluation requires a route analysis (router selections)")
    _items, sealed_items_path = _sealed_split_items(args.data_manifest, digest)
    methods = _sealed_methods(args.run_dir, fault.fault_id)
    if "B0" not in methods:
        raise ValueError(f"sealed evaluation requires a B0 validation record for {fault.fault_id}")
    from neuroforge.artifacts import tree_sha256 as _tree_sha256
    from neuroforge.isolation import file_sha256 as _file_sha256

    outputs: list[str] = []
    sealed_metrics: dict[str, Metrics] = {}
    sealed_records: dict[str, list[dict[str, Any]]] = {}
    targets = (
        ["base"]
        + [f"B0:{fault.fault_id}"]
        + [f"{method}:{fault.fault_id}" for method in methods if method != "B0"]
    )
    base_model = Path("models/tiny-aya-global")
    for target in targets:
        reason = _can_start(args.deadline)
        if reason is not None:
            return _record_stage(
                args.run_dir,
                "sealed",
                _result(
                    status="inconclusive",
                    config_hash=digest,
                    libraries=libraries,
                    started=started,
                    warnings=[reason],
                    outputs=outputs,
                ),
                fault_id=fault.fault_id,
            )
        if target == "base":
            request_doc = _sealed_request_doc(config, digest, "base")
            result = _run_isolated_eval(
                job_name=f"sealed-{fault.fault_id}-base",
                target="base",
                request_doc=request_doc,
                fault_adapter=None,
                candidate_root=None,
                freeze_proof=None,
                base_model=base_model,
                guardian_model=_guardian_path(),
                thresholds_root=args.config.parent,
                sealed_items_path=sealed_items_path,
                config_path=args.config,
                router_selection_path=analysis_path,
            )
            sealed_metrics["base"] = result["metrics"]
            sealed_records["base"] = result["score_records"]
            continue
        method = target.split(":")[0]
        adapter = args.run_dir / "faults" / fault.fault_id / "adapter"
        candidate_root = args.run_dir / "repairs" / fault.fault_id / method
        proof = {
            "candidate_sha256": _tree_sha256(candidate_root),
            "config_sha256": _file_sha256(args.config),
            "router_selection_sha256": _file_sha256(analysis_path),
        }
        request_doc = _sealed_request_doc(config, digest, "candidate")
        result = _run_isolated_eval(
            job_name=f"sealed-{fault.fault_id}-{method}",
            target="candidate",
            request_doc=request_doc,
            fault_adapter=adapter,
            candidate_root=candidate_root,
            freeze_proof=proof,
            base_model=base_model,
            guardian_model=_guardian_path(),
            thresholds_root=args.config.parent,
            sealed_items_path=sealed_items_path,
            config_path=args.config,
            router_selection_path=analysis_path,
        )
        sealed_metrics[method] = result["metrics"]
        sealed_records[method] = result["score_records"]
    base = sealed_metrics["base"]
    drifted = sealed_metrics["B0"]
    details: dict[str, Any] = {"targets": targets, "decisions": {}}
    for method in methods:
        manifest_path = args.run_dir / "repairs" / fault.fault_id / method / "manifest.json"
        candidate_manifest = read_manifest(manifest_path, root=args.run_dir)
        decision = decide(
            base,
            drifted,
            sealed_metrics[method],
            config.acceptance,
            fault_type=fault.fault_type,
            reloadable=bool(getattr(candidate_manifest, "reload_proof", None)),
            integrity_ok=True,
            truncated_to_empty=False,
        )
        record = CandidateRecord(
            candidate_id=f"{fault.fault_id}:{method}",
            fault_id=fault.fault_id,
            method=method,  # type: ignore[arg-type]
            artifact_path=str(manifest_path),
            artifact_sha256=candidate_manifest.artifact_sha256,
            parameters={
                **candidate_manifest.parameters,
                "source_manifest_sha256": sha256_file(manifest_path),
                "sealed_score_records": f"{len(sealed_records[method])} items, no response text",
                "cost_status": "not measured",
            },
            metrics=sealed_metrics[method],
            decision=decision,
            cost_usd=Decimal("0"),
            wall_seconds=time.monotonic() - started,
            split="sealed",
            seed=fault.seed,
            changed_parameter_count=candidate_manifest.changed_parameter_count,
        )
        out_dir = args.run_dir / "sealed" / fault.fault_id / method
        metrics_path = out_dir / "sealed-metrics.json"
        _write_json(
            metrics_path,
            {
                "split": "sealed",
                "metrics": sealed_metrics[method].model_dump(mode="json"),
                "score_records": sealed_records[method],
                "decision": decision.model_dump(mode="json"),
            },
            args.run_dir,
        )
        _write_json(
            out_dir / "sealed-record.json",
            {"config_hash": digest, "candidate": record.model_dump(mode="json")},
            args.run_dir,
        )
        outputs.append(_file_id(metrics_path))
        details["decisions"][method] = decision.model_dump(mode="json")
    return _record_stage(
        args.run_dir,
        "sealed",
        _result(
            status="success",
            config_hash=digest,
            libraries=libraries,
            started=started,
            outputs=outputs,
        ),
        fault_id=fault.fault_id,
        details=details,
    )


def _dispatch(
    args: argparse.Namespace, config: Any, digest: str, libraries: dict[str, str]
) -> StageResult:
    handlers: dict[str, Callable[[argparse.Namespace, Any, str, dict[str, str]], StageResult]] = {
        "baseline": _phase_baseline,
        "first-proof": _phase_first_proof,
        "population": _phase_population,
        "diagnostics": _phase_diagnostics,
        "candidates": _phase_candidates,
        "route": _phase_route,
        "sealed": _phase_sealed,
    }
    return handlers[args.phase](args, config, digest, libraries)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "baseline",
            "first-proof",
            "population",
            "diagnostics",
            "candidates",
            "route",
            "sealed",
        ),
    )
    parser.add_argument("--deadline", type=_parse_deadline)
    parser.add_argument("--hard-budget", type=_parse_budget)
    parser.add_argument("--fault-id")
    args = parser.parse_args(argv)
    if not args.config.is_file() or not args.data_manifest.is_file():
        parser.error("--config and --data-manifest must name existing files")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    try:
        config = _load_config(args.config)
        if args.fault_id is not None:
            _configured_fault(config, args.fault_id)
        digest, libraries = _run_snapshot(config, args.run_dir)
        existing = _existing_stage(args.run_dir, args.phase, digest, fault_id=_stage_key(args))
        if existing is not None:
            result = existing
        else:
            spent = _recorded_candidate_cost(args.run_dir)
            if args.hard_budget is not None and spent >= args.hard_budget:
                result = _record_stage(
                    args.run_dir,
                    args.phase,
                    _result(
                        status="inconclusive",
                        config_hash=digest,
                        libraries=libraries,
                        started=time.monotonic(),
                        warnings=[
                            f"hard-budget guard prevented a new stage; recorded_candidate_cost_usd={spent}",
                            "costs are durable records only; no provider API was queried",
                        ],
                    ),
                    fault_id=_stage_key(args),
                )
            else:
                result = _dispatch(args, config, digest, libraries)
    except Exception as error:
        try:
            config_hash = _run_snapshot(_load_config(args.config), args.run_dir)[0]
            result = _result(
                status="error",
                config_hash=config_hash,
                libraries={},
                started=time.monotonic(),
                errors=[f"{type(error).__name__}: {error}"],
            )
            _record_stage(args.run_dir, f"{args.phase}-error", result, fault_id=args.fault_id)
        except Exception:
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
            return 1
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    return 1 if result.status == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
