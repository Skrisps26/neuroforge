"""Deterministic mocked unit tests; these do not load or run Tiny Aya."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from neuroforge.artifacts import read_json_artifact, write_json_artifact
from neuroforge.contracts import GateOutcome, Metrics, StageResult


@pytest.fixture(scope="module")
def runner_module():
    path = Path("scripts/run_study.py")
    spec = importlib.util.spec_from_file_location("neuroforge_study_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metrics(*, safety: float, refusal: float) -> Metrics:
    return Metrics(
        safety_failure_rate=safety,
        overrefusal_rate=refusal,
        utility=1.0,
        n_safety=1,
        n_benign=1,
        n_utility=1,
        complete=True,
    )


def test_resume_rejects_stage_config_hash_mismatch(tmp_path: Path, runner_module) -> None:
    result = StageResult(
        status="success",
        input_artifact_ids=[],
        output_artifact_ids=[],
        config_hash="a" * 64,
        library_versions={},
        timing={"wall_seconds": 1.0},
    )
    stage = tmp_path / "study-execution" / "baseline.json"
    write_json_artifact(stage, {"phase": "baseline", "fault_id": None, "result": result.model_dump(mode="json"), "details": {}}, root=tmp_path)

    with pytest.raises(ValueError, match="different configuration hash"):
        runner_module._existing_stage(tmp_path, "baseline", "b" * 64)


def test_null_fault_requires_observed_strict_gap(runner_module) -> None:
    f1 = SimpleNamespace(fault_type="F1")
    f2 = SimpleNamespace(fault_type="F2")
    base = _metrics(safety=0.2, refusal=0.1)

    assert runner_module._observed_eligibility(f1, base, _metrics(safety=0.2, refusal=0.1)) == (
        False,
        "null fault: no observed strict safety failure increase",
    )
    assert runner_module._observed_eligibility(f2, base, _metrics(safety=0.2, refusal=0.1)) == (
        False,
        "null fault: no observed strict overrefusal increase",
    )


def test_failed_candidate_is_persisted_without_inferred_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner_module) -> None:
    fault = SimpleNamespace(fault_id="f1-01", fault_type="F1", seed=41)
    failure = StageResult(
        status="error",
        input_artifact_ids=[],
        output_artifact_ids=["real-error-artifact"],
        config_hash="a" * 64,
        library_versions={},
        timing={"wall_seconds": 1.0},
        errors=["RepairUnavailable: upstream rejected the condition"],
    )
    monkeypatch.setattr(runner_module, "_load_curated_split", lambda *_: ([], Path("development.jsonl")))
    monkeypatch.setattr("neuroforge.repairs.repair_candidate", lambda *args, **kwargs: failure)

    record = runner_module._candidate(
        tmp_path,
        SimpleNamespace(),
        fault,
        "M1",
        _metrics(safety=0.1, refusal=0.1),
        _metrics(safety=0.4, refusal=0.1),
        {},
        tmp_path / "manifest.json",
        "a" * 64,
        None,
    )

    persisted = read_json_artifact(tmp_path / "repairs" / "f1-01" / "M1" / "validation-record.json", root=tmp_path)
    assert record.metrics.complete is False
    assert record.decision is not None and record.decision.decision == GateOutcome.ERROR
    assert persisted["candidate"]["decision"]["decision"] == "ERROR"
    assert "upstream rejected" in persisted["candidate"]["decision"]["reasons"][0]


def test_deadline_does_not_start_a_new_candidate_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner_module) -> None:
    fault = SimpleNamespace(fault_id="f1-01", fault_type="F1", seed=41)

    def must_not_load(*args, **kwargs):
        raise AssertionError("deadline guard started a bounded candidate stage")

    monkeypatch.setattr("neuroforge.repairs.repair_candidate", must_not_load)
    record = runner_module._candidate(
        tmp_path,
        SimpleNamespace(),
        fault,
        "M1",
        _metrics(safety=0.1, refusal=0.1),
        _metrics(safety=0.4, refusal=0.1),
        {},
        tmp_path / "manifest.json",
        "a" * 64,
        datetime.now(UTC) - timedelta(seconds=1),
    )

    assert record.decision is not None and record.decision.decision == GateOutcome.INCONCLUSIVE
    assert not (tmp_path / "repairs" / "f1-01" / "M1" / "validation-record.json").exists()
