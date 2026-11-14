from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from neuroforge.contracts import (
    CandidateRecord,
    CloudPreflight,
    CostEvent,
    DiagnosticFeatures,
    GateDecision,
    GateOutcome,
    Metrics,
    PatchManifest,
    RunConfig,
    StageResult,
    canonical_json,
    config_hash,
    load_config,
)

STUDY_CONFIG = Path(__file__).parents[1] / "configs" / "study.yaml"


def _cloud_preflight(**overrides: object) -> CloudPreflight:
    values: dict[str, object] = {
        "purpose": "test",
        "gpu_type": "NVIDIA A40",
        "vram_gb": 48.0,
        "quoted_at": datetime.now(UTC),
        "price_per_hour": Decimal("0.59"),
        "estimated_runtime_seconds": 60.0,
        "hard_runtime_seconds": 60.0,
        "compute_cost": Decimal("0.59"),
        "storage_cost": Decimal("0"),
        "other_cost": Decimal("0"),
        "current_spend": Decimal("0"),
        "projected_spend": Decimal("0.59"),
        "output_location": "runs/test",
        "cleanup_plan": "delete test resources",
    }
    values.update(overrides)
    return CloudPreflight.model_validate(values)


def test_unknown_contract_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Metrics(unexpected_score=0.2)

    with pytest.raises(ValidationError):
        GateDecision(decision="ACCEPT", unreviewed=True)


def test_metric_fractions_are_strict_finite_and_bounded() -> None:
    for invalid in ("0.2", 0, float("nan"), float("inf"), -0.01, 1.01):
        with pytest.raises(ValidationError):
            Metrics(safety_failure_rate=invalid)

    metrics = Metrics(
        safety_failure_rate=0.0,
        overrefusal_rate=1.0,
        utility=0.75,
    )
    assert metrics.safety_failure_rate == 0.0
    assert metrics.overrefusal_rate == 1.0


def test_gate_recoveries_are_finite_but_not_fraction_bounded() -> None:
    decision = GateDecision(
        decision=GateOutcome.ACCEPT,
        safety_recovery=1.5,
        overrefusal_recovery=-0.2,
    )

    restored = GateDecision.model_validate_json(decision.model_dump_json())
    assert restored.safety_recovery == 1.5
    assert restored.overrefusal_recovery == -0.2

    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError):
            GateDecision(decision=GateOutcome.ACCEPT, safety_recovery=value)


def test_metrics_reject_negative_sample_counts() -> None:
    for field in ("n_safety", "n_benign", "n_utility"):
        with pytest.raises(ValidationError):
            Metrics(**{field: -1})


def test_durable_numeric_fields_reject_nonfinite_values() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError):
            StageResult(
                status="success",
                input_artifact_ids=[],
                output_artifact_ids=[],
                config_hash="f" * 64,
                library_versions={},
                timing={"wall_seconds": value},
            )

        with pytest.raises(ValidationError):
            DiagnosticFeatures(
                fault_id="f1-01",
                fault_type="F1",
                stable_rank=value,
                lora_rank=4,
            )

        for field in (
            "vram_gb",
            "estimated_runtime_seconds",
            "hard_runtime_seconds",
        ):
            with pytest.raises(ValidationError):
                _cloud_preflight(**{field: value})

    for value in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
        with pytest.raises(ValidationError):
            _cloud_preflight(price_per_hour=value)


@pytest.mark.parametrize("duration", [0.0, -1.0])
def test_stage_timings_must_be_positive(duration: float) -> None:
    with pytest.raises(ValidationError):
        StageResult(
            status="success",
            input_artifact_ids=[],
            output_artifact_ids=[],
            config_hash="f" * 64,
            library_versions={},
            timing={"wall_seconds": duration},
        )


def test_missing_metrics_remain_distinct_from_measured_zero() -> None:
    missing = Metrics()
    measured_zero = Metrics(safety_failure_rate=0.0)

    assert missing.safety_failure_rate is None
    assert measured_zero.safety_failure_rate == 0.0
    assert missing.model_dump() != measured_zero.model_dump()


def test_patch_manifest_rejects_invalid_artifact_hash() -> None:
    decision = GateDecision(decision=GateOutcome.ACCEPT)
    metrics = Metrics(safety_failure_rate=0.1, complete=True)
    common = {
        "base_model": "CohereLabs/tiny-aya-global",
        "base_revision": "revision",
        "fault_id": "f1-01",
        "method": "M1",
        "artifact_path": "patches/f1-01.safetensors",
        "validation_metrics": metrics,
        "decision": decision,
        "rollback_reference": "faults/f1-01",
    }

    with pytest.raises(ValidationError):
        PatchManifest(artifact_sha256="not-a-sha256", **common)

    manifest = PatchManifest(artifact_sha256="a" * 64, **common)
    assert manifest.schema_version == 1


def test_stage_errors_survive_json_serialization() -> None:
    stage = StageResult(
        status="error",
        input_artifact_ids=["inputs-1"],
        output_artifact_ids=[],
        config_hash="f" * 64,
        library_versions={"pydantic": "2"},
        timing={"wall_seconds": 1.0},
        errors=["scorer unavailable"],
    )

    restored = StageResult.model_validate_json(stage.model_dump_json())
    assert restored.status == "error"
    assert restored.errors == ["scorer unavailable"]


def test_canonical_json_and_hash_are_order_independent() -> None:
    first = {"b": Decimal("1.50"), "a": [2, 1]}
    second = {"a": [2, 1], "b": Decimal("1.50")}

    assert canonical_json(first) == canonical_json(second)
    assert config_hash(first) == config_hash(second)


def test_cost_event_rejects_negative_amount() -> None:
    with pytest.raises(ValidationError):
        CostEvent(
            event_id="reserve-1",
            event_type="reserve",
            resource_id="pod-1",
            amount_usd=Decimal("-0.01"),
            timestamp=datetime.now(UTC),
            purpose="test",
        )


def test_candidate_requires_known_method_and_nonnegative_cost() -> None:
    metrics = Metrics(safety_failure_rate=0.2)
    with pytest.raises(ValidationError):
        CandidateRecord(
            candidate_id="candidate-1",
            fault_id="f1-01",
            method="unknown",
            metrics=metrics,
            cost_usd=Decimal("0"),
            wall_seconds=0.0,
            split="validation",
            seed=42,
        )


def test_study_yaml_validates_and_round_trips(tmp_path: Path) -> None:
    config = load_config(STUDY_CONFIG)
    assert config.model_id == "CohereLabs/tiny-aya-global"
    assert config.model_revision == "af89d219b53ed9b13b8a4645f9c8028973510324"
    assert [fault.fault_id for fault in config.faults] == [
        "f1-01",
        "f1-02",
        "f1-03",
        "f2-01",
        "f2-02",
        "f2-03",
    ]
    assert config.split_sizes == {"development": 20, "validation": 30, "sealed": 50}

    with pytest.raises(ValidationError):
        RunConfig.model_validate({**config.model_dump(), "unexpected": True})

    round_trip_path = tmp_path / "study.yaml"
    round_trip_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    assert load_config(round_trip_path) == config
