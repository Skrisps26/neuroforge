from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from neuroforge.contracts import CandidateRecord, DiagnosticFeatures, GateDecision, Metrics
from neuroforge.router import (
    ESCALATE,
    _fit_router,
    _resolve_choice,
    analyze_router,
    RouterInputError,
)


def _metrics(*, safety: float = 0.8, overrefusal: float = 0.4, utility: float = 0.8) -> Metrics:
    return Metrics(
        safety_failure_rate=safety,
        overrefusal_rate=overrefusal,
        utility=utility,
        n_safety=10,
        n_benign=10,
        n_utility=10,
        complete=True,
    )


def _candidate(
    fault_id: str,
    method: str,
    score: float,
    *,
    accepted: bool = True,
    complete: bool = True,
    split: str = "validation",
    seed: int = 42,
) -> CandidateRecord:
    metrics = _metrics(safety=0.8 - score)
    if not complete:
        metrics = Metrics(safety_failure_rate=metrics.safety_failure_rate, complete=False)
    return CandidateRecord(
        candidate_id=f"{fault_id}-{method}-{seed}",
        fault_id=fault_id,
        method=method,
        metrics=metrics,
        decision=GateDecision(decision="ACCEPT" if accepted else "REJECT"),
        cost_usd=Decimal("0.10"),
        wall_seconds=1.0,
        split=split,
        seed=seed,
    )


def _inputs(*, include_random: bool = False):
    fault_ids = [f"f{i}" for i in range(6)]
    features = [
        DiagnosticFeatures(
            fault_id=fault_id,
            fault_type="F1" if index < 3 else "F2",
            stable_rank=float(index + 1),
            top8_energy=0.2,
            patching_sharpness=0.6,
            relative_drift_norm=0.1,
            lora_rank=4,
            circuit_concentration=0.7,
        )
        for index, fault_id in enumerate(fault_ids)
    ]
    candidates = []
    for index, fault_id in enumerate(fault_ids):
        candidates.extend(
            [
                _candidate(fault_id, "M1", 0.1 + (index % 2) * 0.1),
                _candidate(fault_id, "M2", 0.2 - (index % 2) * 0.1),
                _candidate(fault_id, "M3", 0.05),
            ]
        )
        if include_random:
            candidates.extend(
                [
                    _candidate(fault_id, "R0", 0.1),
                    _candidate(fault_id, "R1", 0.2),
                    _candidate(fault_id, "R2", 0.3),
                ]
            )
    drift = {fault_id: _metrics() for fault_id in fault_ids}
    return features, candidates, drift, fault_ids


def test_fixed_baseline_is_selected_inside_each_training_fold() -> None:
    features, candidates, drift, fault_ids = _inputs()
    candidates = [candidate for candidate in candidates if candidate.fault_id != "f0"]
    candidates.extend(
        [
            _candidate("f0", "M1", 0.79),
            _candidate("f0", "M2", 0.13),
            _candidate("f0", "M3", 0.0),
        ]
    )
    for fault_id in fault_ids[1:]:
        candidates = [
            candidate
            for candidate in candidates
            if not (candidate.fault_id == fault_id and candidate.method in {"M1", "M2"})
        ]
        candidates.extend([_candidate(fault_id, "M1", 0.0), _candidate(fault_id, "M2", 0.13)])

    result = analyze_router(features, candidates, drift, observed_fault_ids=fault_ids)

    held_out = next(
        record
        for record in result["leave_one_checkpoint_out"]["records"]
        if record["fault_id"] == "f0"
    )
    assert held_out["fixed"]["requested_method"] == "M2"
    assert result["hindsight_fixed"]["method"] == "M1"


def test_fold_imputer_does_not_learn_held_out_nonmissing_values() -> None:
    train = np.asarray([[np.nan, 1.0], [np.nan, 2.0]])
    held_out = np.asarray([[999.0, 3.0]])
    model = _fit_router(train, [ESCALATE, ESCALATE])
    imputer = model.named_steps["impute"]

    assert isinstance(model.named_steps["router"], DummyClassifier)
    np.testing.assert_array_equal(imputer.transform(train), [[0.0, 1.0], [0.0, 2.0]])
    np.testing.assert_array_equal(imputer.transform(held_out), held_out)
    assert model.predict(held_out).tolist() == [ESCALATE]


def test_negative_passing_repair_keeps_negative_score_and_nonnegative_regret() -> None:
    features, candidates, drift, fault_ids = _inputs()
    candidates = [
        _candidate(candidate.fault_id, candidate.method, -0.1 if candidate.method == "M1" else 0.0)
        for candidate in candidates
    ]
    result = analyze_router(features, candidates, drift, observed_fault_ids=fault_ids)
    first_fault = fault_ids[0]
    candidate_map = {
        candidate.method: candidate for candidate in candidates if candidate.fault_id == first_fault
    }

    selected, score, passes, _ = _resolve_choice("M1", first_fault, {first_fault: candidate_map}, drift[first_fault])
    assert selected == "M1"
    assert score == pytest.approx(-0.1)
    assert passes
    for record in result["leave_one_checkpoint_out"]["records"]:
        assert record["oracle"]["label"] == ESCALATE
        assert record["oracle"]["operational_score"] == 0.0
        assert all(record[name]["regret"] >= 0.0 for name in ("router", "fixed", "random"))


def test_sealed_unknown_and_duplicate_candidates_are_rejected() -> None:
    features, candidates, drift, fault_ids = _inputs()
    sealed = _candidate("f0", "M1", 0.1, split="sealed")
    with pytest.raises(RouterInputError, match="sealed"):
        analyze_router(features, [*candidates, sealed], drift, observed_fault_ids=fault_ids)

    unknown = _candidate("unknown", "M1", 0.1)
    with pytest.raises(RouterInputError, match="unknown"):
        analyze_router(features, [*candidates, unknown], drift, observed_fault_ids=fault_ids)

    duplicate = _candidate("f0", "M1", 0.1)
    with pytest.raises(RouterInputError, match="duplicate"):
        analyze_router(features, [*candidates, duplicate], drift, observed_fault_ids=fault_ids)


def test_partially_measured_wsp_is_not_a_router_arm() -> None:
    features, candidates, drift, fault_ids = _inputs()
    candidates.extend(
        _candidate(fault_id, "W1", 0.1, complete=fault_id != "f0") for fault_id in fault_ids
    )

    result = analyze_router(features, candidates, drift, observed_fault_ids=fault_ids)

    assert "W1" not in result["cohort"]["common_methods"]


def test_fault_type_folds_are_deterministic_and_train_on_other_type() -> None:
    features, candidates, drift, fault_ids = _inputs()
    first = analyze_router(features, candidates, drift, observed_fault_ids=fault_ids)
    second = analyze_router(features, candidates, drift, observed_fault_ids=fault_ids)

    assert first["leave_one_fault_type_out"] == second["leave_one_fault_type_out"]
    for record in first["leave_one_fault_type_out"]["records"]:
        held_type = next(feature.fault_type for feature in features if feature.fault_id == record["fault_id"])
        assert all(
            next(feature.fault_type for feature in features if feature.fault_id == train_id) != held_type
            for train_id in record["train_fault_ids"]
        )


def test_bootstrap_reports_checkpoint_units() -> None:
    features, candidates, drift, fault_ids = _inputs()
    result = analyze_router(features, candidates, drift, observed_fault_ids=fault_ids)
    interval = result["leave_one_checkpoint_out"]["bootstrap"]["router_vs_random_regret"]

    assert interval["unit"] == "checkpoint"
    assert interval["n_resamples"] == 10_000


def test_h3_matches_m2_to_mean_of_its_three_random_draws_without_prompt_pooling() -> None:
    features, candidates, drift, fault_ids = _inputs(include_random=True)
    result = analyze_router(features, candidates, drift, observed_fault_ids=fault_ids[:2])

    assert result["status"] == "insufficient"
    h3_record = result["h3"]["records"][0]
    assert h3_record["random_scores"] == pytest.approx([0.1, 0.2, 0.3])
    assert h3_record["mean_random_score"] == pytest.approx(0.2)
    assert h3_record["m2_minus_mean_random"] == pytest.approx(0.0)


def test_missing_observed_status_is_honest_insufficiency() -> None:
    features, candidates, drift, _ = _inputs()

    result = analyze_router(features, candidates, drift)

    assert result["status"] == "insufficient"
    assert "explicit observed_fault_ids" in result["reasons"][0]
