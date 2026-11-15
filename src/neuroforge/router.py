"""Validation-only router analysis for measured NeuroForge repair candidates.

WHY CUSTOM: scikit-learn supplies the prescribed imputer, pipeline, and classifiers and
SciPy supplies the bootstrap.  Neither library encodes this study's sealed-data barrier,
measured-arm cohort rule, gate semantics, or preregistered score and tie-breaking rules.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from decimal import Decimal
from typing import Any

import numpy as np
from scipy.stats import bootstrap
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .contracts import CandidateRecord, DiagnosticFeatures, GateOutcome, Metrics

FEATURE_NAMES = (
    "stable_rank",
    "top8_energy",
    "patching_sharpness",
    "relative_drift_norm",
    "lora_rank",
)
CORE_METHODS = ("M1", "M2", "M3")
OPTIONAL_METHODS = ("M4", "W1")
RANDOM_CONTROL_METHODS = ("R0", "R1", "R2")
ESCALATE = "escalate"


class RouterInputError(ValueError):
    """Raised when inputs cannot support a leakage-safe router analysis."""


def analyze_router(
    features: Sequence[DiagnosticFeatures],
    candidates: Sequence[CandidateRecord],
    drift_metrics: Mapping[str, Metrics],
    *,
    observed_fault_ids: Collection[str] | None = None,
    base_metrics: Mapping[str, Metrics] | None = None,
    diagnostic_cost_usd: Mapping[str, Decimal | float] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Analyze validation candidates without deriving fault eligibility from recipes.

    ``observed_fault_ids`` is an explicit, measured status supplied by the evaluator.  It
    must not be inferred from a fault's intended recipe or its candidate list.  The optional
    ``base_metrics`` completes the measured base/drift API and is retained for the caller's
    provenance; gate outcomes, rather than a reimplemented gate, determine eligibility.
    """
    feature_by_fault = _index_features(features)
    candidate_by_fault = _index_candidates(candidates, feature_by_fault, drift_metrics)
    _validate_metric_mapping(drift_metrics, "drift_metrics")
    if base_metrics is not None:
        _validate_metric_mapping(base_metrics, "base_metrics")

    if observed_fault_ids is None:
        return _insufficient(
            (),
            (),
            ["explicit observed_fault_ids are required; intended recipes are not evidence"],
            _untested_h3("explicit observed fault status was not supplied"),
            base_metrics is not None,
        )

    fault_ids = tuple(sorted(set(observed_fault_ids)))
    if not fault_ids:
        return _insufficient(
            (),
            (),
            ["no observed faults were supplied"],
            _untested_h3("no observed faults were supplied"),
            base_metrics is not None,
        )
    unknown = [fault_id for fault_id in fault_ids if fault_id not in feature_by_fault]
    if unknown:
        raise RouterInputError(f"observed faults have no diagnostics: {', '.join(unknown)}")
    unknown = [fault_id for fault_id in fault_ids if fault_id not in drift_metrics]
    if unknown:
        raise RouterInputError(f"observed faults have no drift metrics: {', '.join(unknown)}")
    if base_metrics is not None:
        unknown = [fault_id for fault_id in fault_ids if fault_id not in base_metrics]
        if unknown:
            raise RouterInputError(f"observed faults have no base metrics: {', '.join(unknown)}")

    fault_types = tuple(feature_by_fault[fault_id].fault_type for fault_id in fault_ids)
    h3 = _analyze_h3(fault_ids, feature_by_fault, candidate_by_fault, drift_metrics)
    methods = _common_methods(fault_ids, candidate_by_fault)
    reasons: list[str] = []
    if len(fault_ids) < 6:
        reasons.append("router untested: fewer than six observed checkpoints")
    if len(set(fault_types)) < 2:
        reasons.append("router untested: fewer than two observed intended fault types")
    if not methods:
        reasons.append("router untested: no operational repair is completely measured on every fault")
    if reasons:
        return _insufficient(fault_ids, fault_types, reasons, h3, base_metrics is not None, methods)

    matrix = _feature_matrix(fault_ids, feature_by_fault)
    labels = tuple(
        _oracle_label(fault_id, methods, candidate_by_fault, drift_metrics[fault_id])
        for fault_id in fault_ids
    )
    loo = _evaluate_folds(
        "leave_one_checkpoint_out",
        ((index,) for index in range(len(fault_ids))),
        fault_ids,
        matrix,
        labels,
        methods,
        candidate_by_fault,
        drift_metrics,
        diagnostic_cost_usd,
        seed,
    )
    lfo = _evaluate_folds(
        "leave_one_fault_type_out",
        (
            tuple(index for index, fault_type in enumerate(fault_types) if fault_type == held_out_type)
            for held_out_type in sorted(set(fault_types))
        ),
        fault_ids,
        matrix,
        labels,
        methods,
        candidate_by_fault,
        drift_metrics,
        diagnostic_cost_usd,
        seed,
    )
    hindsight_method = _best_fixed_method(
        tuple(range(len(fault_ids))), fault_ids, methods, candidate_by_fault, drift_metrics
    )
    hindsight = _evaluate_fixed_method(
        hindsight_method,
        fault_ids,
        labels,
        methods,
        candidate_by_fault,
        drift_metrics,
        diagnostic_cost_usd,
    )

    return {
        "status": "success",
        "cohort": {
            "fault_ids": list(fault_ids),
            "fault_types": list(fault_types),
            "n_checkpoints": len(fault_ids),
            "common_methods": list(methods),
            "base_metrics_supplied": base_metrics is not None,
        },
        "feature_names": list(FEATURE_NAMES),
        "labels": {fault_id: label for fault_id, label in zip(fault_ids, labels, strict=True)},
        "leave_one_checkpoint_out": loo,
        "leave_one_fault_type_out": lfo,
        "hindsight_fixed": {
            "method": hindsight_method,
            "warning": "retrospective best fixed method; not a cross-validation baseline",
            **hindsight,
        },
        "h3": h3,
    }


def _index_features(features: Sequence[DiagnosticFeatures]) -> dict[str, DiagnosticFeatures]:
    indexed: dict[str, DiagnosticFeatures] = {}
    for feature in features:
        if feature.fault_id in indexed:
            raise RouterInputError(f"duplicate diagnostics for fault {feature.fault_id}")
        indexed[feature.fault_id] = feature
    return indexed


def _index_candidates(
    candidates: Sequence[CandidateRecord],
    feature_by_fault: Mapping[str, DiagnosticFeatures],
    drift_metrics: Mapping[str, Metrics],
) -> dict[str, dict[str, CandidateRecord]]:
    indexed: dict[str, dict[str, CandidateRecord]] = {}
    identities: set[tuple[str, str, int]] = set()
    for candidate in candidates:
        if candidate.split == "sealed":
            raise RouterInputError("sealed candidates are forbidden in router analysis")
        if candidate.split != "validation":
            raise RouterInputError("router candidates must be validation-only")
        if candidate.fault_id not in feature_by_fault or candidate.fault_id not in drift_metrics:
            raise RouterInputError(f"candidate references unknown fault {candidate.fault_id}")
        identity = (candidate.fault_id, candidate.method, candidate.seed)
        if identity in identities:
            raise RouterInputError(
                "duplicate fault/method/seed candidate identity: " + "/".join(map(str, identity))
            )
        identities.add(identity)
        methods = indexed.setdefault(candidate.fault_id, {})
        if candidate.method in methods:
            raise RouterInputError(
                f"ambiguous multiple candidates for {candidate.fault_id}/{candidate.method}"
            )
        methods[candidate.method] = candidate
    return indexed


def _validate_metric_mapping(metrics_by_fault: Mapping[str, Metrics], name: str) -> None:
    for fault_id, metrics in metrics_by_fault.items():
        if not isinstance(fault_id, str) or not isinstance(metrics, Metrics):
            raise RouterInputError(f"{name} must map fault IDs to Metrics")


def _feature_matrix(
    fault_ids: Sequence[str], feature_by_fault: Mapping[str, DiagnosticFeatures]
) -> np.ndarray:
    return np.asarray(
        [
            [
                np.nan if getattr(feature_by_fault[fault_id], name) is None else getattr(feature_by_fault[fault_id], name)
                for name in FEATURE_NAMES
            ]
            for fault_id in fault_ids
        ],
        dtype=float,
    )


def _is_complete(metrics: Metrics) -> bool:
    return (
        metrics.complete
        and not metrics.evaluation_errors
        and metrics.safety_failure_rate is not None
        and metrics.overrefusal_rate is not None
        and metrics.utility is not None
    )


def _is_measured(candidate: CandidateRecord | None) -> bool:
    return candidate is not None and candidate.decision is not None and _is_complete(candidate.metrics)


def _candidate_score(candidate: CandidateRecord | None, drift: Metrics) -> float | None:
    if candidate is None or not _is_complete(candidate.metrics) or not _is_complete(drift):
        return None
    return (
        drift.safety_failure_rate
        - candidate.metrics.safety_failure_rate
        + drift.overrefusal_rate
        - candidate.metrics.overrefusal_rate
        - max(0.0, drift.utility - candidate.metrics.utility)
    )


def _passing_score(candidate: CandidateRecord | None, drift: Metrics) -> float | None:
    if candidate is None or candidate.decision is None or candidate.decision.decision != GateOutcome.ACCEPT:
        return None
    return _candidate_score(candidate, drift)


def _common_methods(
    fault_ids: Sequence[str], candidate_by_fault: Mapping[str, Mapping[str, CandidateRecord]]
) -> tuple[str, ...]:
    methods: list[str] = []
    for method in (*CORE_METHODS, *OPTIONAL_METHODS):
        if all(_is_measured(candidate_by_fault.get(fault_id, {}).get(method)) for fault_id in fault_ids):
            methods.append(method)
    return tuple(methods)


def _candidate_sort_key(candidate: CandidateRecord) -> tuple[float, float, str]:
    return (float(candidate.cost_usd), candidate.wall_seconds, candidate.method)


def _oracle_label(
    fault_id: str,
    methods: Sequence[str],
    candidate_by_fault: Mapping[str, Mapping[str, CandidateRecord]],
    drift: Metrics,
) -> str:
    choices = [
        (method, _passing_score(candidate_by_fault[fault_id][method], drift)) for method in methods
    ]
    passing = [(method, score) for method, score in choices if score is not None]
    if not passing:
        return ESCALATE
    best_score = max(score for _, score in passing)
    if best_score <= 0.0:
        return ESCALATE
    tied = [
        candidate_by_fault[fault_id][method]
        for method, score in passing
        if score == best_score
    ]
    return min(tied, key=_candidate_sort_key).method


def _oracle_scores(
    fault_id: str,
    methods: Sequence[str],
    candidate_by_fault: Mapping[str, Mapping[str, CandidateRecord]],
    drift: Metrics,
) -> tuple[float, float | None]:
    scores = [
        score
        for method in methods
        if (score := _passing_score(candidate_by_fault[fault_id][method], drift)) is not None
    ]
    repair_only = max(scores) if scores else None
    return max(0.0, repair_only or 0.0), repair_only


def _resolve_choice(
    requested_method: str,
    fault_id: str,
    candidate_by_fault: Mapping[str, Mapping[str, CandidateRecord]],
    drift: Metrics,
) -> tuple[str, float, bool, CandidateRecord | None]:
    if requested_method == ESCALATE:
        return ESCALATE, 0.0, False, None
    candidate = candidate_by_fault[fault_id].get(requested_method)
    score = _passing_score(candidate, drift)
    if score is None:
        return ESCALATE, 0.0, False, None
    return requested_method, score, True, candidate


def _best_fixed_method(
    train_indices: Sequence[int],
    fault_ids: Sequence[str],
    methods: Sequence[str],
    candidate_by_fault: Mapping[str, Mapping[str, CandidateRecord]],
    drift_metrics: Mapping[str, Metrics],
) -> str:
    ranked: list[tuple[float, float, float, str]] = []
    for method in methods:
        scores: list[float] = []
        costs: list[float] = []
        durations: list[float] = []
        for index in train_indices:
            fault_id = fault_ids[index]
            candidate = candidate_by_fault[fault_id][method]
            raw_score = _passing_score(candidate, drift_metrics[fault_id])
            scores.append(raw_score if raw_score is not None else 0.0)
            costs.append(float(candidate.cost_usd))
            durations.append(candidate.wall_seconds)
        ranked.append((-float(np.mean(scores)), float(np.mean(costs)), float(np.mean(durations)), method))
    return min(ranked)[3]


def _fit_router(train_x: np.ndarray, train_y: Sequence[str]) -> Pipeline:
    classifier: DecisionTreeClassifier | DummyClassifier
    if len(set(train_y)) == 1:
        classifier = DummyClassifier(strategy="most_frequent")
    else:
        classifier = DecisionTreeClassifier(max_depth=2, min_samples_leaf=2, random_state=42)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("router", classifier),
        ]
    ).fit(train_x, train_y)


def _evaluate_folds(
    name: str,
    held_out_groups: Sequence[Sequence[int]] | Any,
    fault_ids: Sequence[str],
    matrix: np.ndarray,
    labels: Sequence[str],
    methods: Sequence[str],
    candidate_by_fault: Mapping[str, Mapping[str, CandidateRecord]],
    drift_metrics: Mapping[str, Metrics],
    diagnostic_cost_usd: Mapping[str, Decimal | float] | None,
    seed: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    random_generator = np.random.default_rng(seed)
    all_indices = tuple(range(len(fault_ids)))
    for fold_number, group in enumerate(held_out_groups):
        held_out = tuple(group)
        train_indices = tuple(index for index in all_indices if index not in held_out)
        if not train_indices:
            raise RouterInputError(f"{name} has an empty training fold")
        model = _fit_router(matrix[list(train_indices)], [labels[index] for index in train_indices])
        fixed_method = _best_fixed_method(
            train_indices, fault_ids, methods, candidate_by_fault, drift_metrics
        )
        predictions = model.predict(matrix[list(held_out)])
        for index, prediction in zip(held_out, predictions, strict=True):
            fault_id = fault_ids[index]
            requested_random = str(random_generator.choice([*methods, ESCALATE]))
            records.append(
                _checkpoint_record(
                    fault_id,
                    labels[index],
                    str(prediction),
                    fixed_method,
                    requested_random,
                    candidate_by_fault,
                    drift_metrics,
                    methods,
                    diagnostic_cost_usd,
                    fold_number,
                    tuple(fault_ids[train_index] for train_index in train_indices),
                )
            )
    records.sort(key=lambda row: row["fault_id"])
    return {
        "scheme": name,
        "records": records,
        "aggregates": _aggregates(records),
        "bootstrap": {
            "router_vs_train_fold_fixed_regret": _bootstrap_regret(records, "fixed"),
            "router_vs_random_regret": _bootstrap_regret(records, "random"),
        },
    }


def _checkpoint_record(
    fault_id: str,
    oracle_label: str,
    router_requested: str,
    fixed_requested: str,
    random_requested: str,
    candidate_by_fault: Mapping[str, Mapping[str, CandidateRecord]],
    drift_metrics: Mapping[str, Metrics],
    methods: Sequence[str],
    diagnostic_cost_usd: Mapping[str, Decimal | float] | None,
    fold: int,
    train_fault_ids: Sequence[str],
) -> dict[str, Any]:
    drift = drift_metrics[fault_id]
    oracle_score, repair_only_score = _oracle_scores(fault_id, methods, candidate_by_fault, drift)
    router = _choice_payload(
        router_requested, oracle_label, fault_id, candidate_by_fault, drift, oracle_score, diagnostic_cost_usd
    )
    fixed = _choice_payload(
        fixed_requested, oracle_label, fault_id, candidate_by_fault, drift, oracle_score, diagnostic_cost_usd
    )
    random_choice = _choice_payload(
        random_requested, oracle_label, fault_id, candidate_by_fault, drift, oracle_score, diagnostic_cost_usd
    )
    return {
        "fault_id": fault_id,
        "fold": fold,
        "train_fault_ids": list(train_fault_ids),
        "oracle": {
            "label": oracle_label,
            "operational_score": oracle_score,
            "best_repair_only_score": repair_only_score,
        },
        "router": router,
        "fixed": fixed,
        "random": random_choice,
    }


def _choice_payload(
    requested: str,
    oracle_label: str,
    fault_id: str,
    candidate_by_fault: Mapping[str, Mapping[str, CandidateRecord]],
    drift: Metrics,
    oracle_score: float,
    diagnostic_cost_usd: Mapping[str, Decimal | float] | None,
) -> dict[str, Any]:
    selected, score, passes, candidate = _resolve_choice(requested, fault_id, candidate_by_fault, drift)
    diagnostic_cost = _diagnostic_cost(fault_id, diagnostic_cost_usd)
    candidate_cost = float(candidate.cost_usd) if candidate is not None else 0.0
    total_cost = None if diagnostic_cost is None else diagnostic_cost + candidate_cost
    return {
        "requested_method": requested,
        "selected_method": selected,
        "passes_gate": passes,
        "operational_score": score,
        "regret": max(0.0, oracle_score - score),
        "matches_oracle_label": selected == oracle_label,
        "candidate_cost_usd": candidate_cost if candidate is not None else None,
        "diagnostic_cost_usd": diagnostic_cost,
        "total_measured_cost_usd": total_cost,
    }


def _diagnostic_cost(
    fault_id: str, diagnostic_cost_usd: Mapping[str, Decimal | float] | None
) -> float | None:
    if diagnostic_cost_usd is None or fault_id not in diagnostic_cost_usd:
        return None
    value = float(diagnostic_cost_usd[fault_id])
    if not np.isfinite(value) or value < 0.0:
        raise RouterInputError(f"diagnostic cost for {fault_id} must be finite and nonnegative")
    return value


def _aggregates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for strategy in ("router", "fixed", "random"):
        choices = [record[strategy] for record in records]
        known_costs = [choice["total_measured_cost_usd"] for choice in choices]
        result[strategy] = {
            "mean_regret": float(np.mean([choice["regret"] for choice in choices])),
            "mean_operational_score": float(np.mean([choice["operational_score"] for choice in choices])),
            "method_accuracy": float(np.mean([choice["matches_oracle_label"] for choice in choices])),
            "gate_pass_rate": float(np.mean([choice["passes_gate"] for choice in choices])),
            "mean_total_measured_cost_usd": (
                float(np.mean(known_costs)) if all(cost is not None for cost in known_costs) else None
            ),
            "cost_comparison": (
                "not claimed: explicit diagnostic cost is missing for one or more checkpoints"
                if any(cost is None for cost in known_costs)
                else "measured selected-candidate cost includes supplied diagnostics cost; no savings claim"
            ),
        }
    return result


def _bootstrap_regret(records: Sequence[Mapping[str, Any]], comparator: str) -> dict[str, Any]:
    router = np.asarray([record["router"]["regret"] for record in records], dtype=float)
    other = np.asarray([record[comparator]["regret"] for record in records], dtype=float)
    difference = router - other
    if len(difference) < 2:
        return {"status": "unestimable", "reason": "fewer than two checkpoint units"}
    if np.all(difference == difference[0]):
        return {
            "status": "degenerate",
            "mean_router_minus_comparator_regret": float(difference[0]),
            "confidence_interval": [float(difference[0]), float(difference[0])],
            "significance_claim": False,
        }

    def paired_mean(first: np.ndarray, second: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.mean(first - second, axis=axis)

    interval = bootstrap(
        (router, other),
        paired_mean,
        paired=True,
        vectorized=True,
        method="percentile",
        n_resamples=10_000,
        confidence_level=0.95,
        rng=42,
    ).confidence_interval
    return {
        "status": "estimated",
        "mean_router_minus_comparator_regret": float(np.mean(difference)),
        "confidence_interval": [float(interval.low), float(interval.high)],
        "unit": "checkpoint",
        "n_resamples": 10_000,
        "significance_claim": False,
    }


def _evaluate_fixed_method(
    method: str,
    fault_ids: Sequence[str],
    labels: Sequence[str],
    methods: Sequence[str],
    candidate_by_fault: Mapping[str, Mapping[str, CandidateRecord]],
    drift_metrics: Mapping[str, Metrics],
    diagnostic_cost_usd: Mapping[str, Decimal | float] | None,
) -> dict[str, Any]:
    records = []
    for fault_id, label in zip(fault_ids, labels, strict=True):
        oracle_score, repair_only_score = _oracle_scores(
            fault_id, methods, candidate_by_fault, drift_metrics[fault_id]
        )
        choice = _choice_payload(
            method,
            label,
            fault_id,
            candidate_by_fault,
            drift_metrics[fault_id],
            oracle_score,
            diagnostic_cost_usd,
        )
        records.append(
            {
                "fault_id": fault_id,
                "oracle_label": label,
                "oracle_operational_score": oracle_score,
                "best_repair_only_score": repair_only_score,
                "fixed": choice,
            }
        )
    return {"records": records, "aggregates": _aggregates_for_fixed(records)}


def _aggregates_for_fixed(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    choices = [record["fixed"] for record in records]
    costs = [choice["total_measured_cost_usd"] for choice in choices]
    return {
        "mean_regret": float(np.mean([choice["regret"] for choice in choices])),
        "mean_operational_score": float(np.mean([choice["operational_score"] for choice in choices])),
        "gate_pass_rate": float(np.mean([choice["passes_gate"] for choice in choices])),
        "mean_total_measured_cost_usd": float(np.mean(costs)) if all(cost is not None for cost in costs) else None,
    }


def _analyze_h3(
    fault_ids: Sequence[str],
    feature_by_fault: Mapping[str, DiagnosticFeatures],
    candidate_by_fault: Mapping[str, Mapping[str, CandidateRecord]],
    drift_metrics: Mapping[str, Metrics],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for fault_id in fault_ids:
        feature = feature_by_fault[fault_id]
        methods = candidate_by_fault.get(fault_id, {})
        if feature.patching_sharpness is None or feature.circuit_concentration is None:
            missing.append(f"{fault_id}: D2/D3 sharpness unavailable")
            continue
        required = ("M2", *RANDOM_CONTROL_METHODS)
        if any(method not in methods for method in required):
            missing.append(f"{fault_id}: M2 and all R0/R1/R2 draws are required")
            continue
        m2_score = _candidate_score(methods["M2"], drift_metrics[fault_id])
        random_scores = [
            _candidate_score(methods[method], drift_metrics[fault_id])
            for method in RANDOM_CONTROL_METHODS
        ]
        if m2_score is None or any(score is None for score in random_scores):
            missing.append(f"{fault_id}: H3 requires complete M2 and R0/R1/R2 metrics")
            continue
        mean_random = float(np.mean(random_scores))
        records.append(
            {
                "fault_id": fault_id,
                "patching_sharpness": feature.patching_sharpness,
                "circuit_concentration": feature.circuit_concentration,
                "m2_score": m2_score,
                "random_scores": random_scores,
                "mean_random_score": mean_random,
                "m2_minus_mean_random": m2_score - mean_random,
                "m2_passes_gate": methods["M2"].decision is not None
                and methods["M2"].decision.decision == GateOutcome.ACCEPT,
                "random_pass_rate": float(
                    np.mean(
                        [
                            methods[method].decision is not None
                            and methods[method].decision.decision == GateOutcome.ACCEPT
                            for method in RANDOM_CONTROL_METHODS
                        ]
                    )
                ),
                "sharpness_group": "high" if feature.patching_sharpness >= 0.5 else "low",
            }
        )
    if not records:
        return _untested_h3("; ".join(missing) if missing else "no H3 measurements")
    groups: dict[str, dict[str, Any]] = {}
    for group in ("high", "low"):
        grouped = [record for record in records if record["sharpness_group"] == group]
        groups[group] = (
            {
                "status": "tested",
                "n_checkpoints": len(grouped),
                "mean_m2_minus_mean_random": float(
                    np.mean([record["m2_minus_mean_random"] for record in grouped])
                ),
            }
            if grouped
            else {"status": "untested", "reason": f"no {group}-sharpness checkpoints"}
        )
    return {
        "status": "tested",
        "records": records,
        "high_sharpness_cutoff": 0.5,
        "strata": groups,
        "limitations": missing,
    }


def _untested_h3(reason: str) -> dict[str, Any]:
    return {"status": "untested", "reason": reason, "high_sharpness_cutoff": 0.5}


def _insufficient(
    fault_ids: Sequence[str],
    fault_types: Sequence[str],
    reasons: Sequence[str],
    h3: Mapping[str, Any],
    base_metrics_supplied: bool,
    methods: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "status": "insufficient",
        "reasons": list(reasons),
        "cohort": {
            "fault_ids": list(fault_ids),
            "fault_types": list(fault_types),
            "n_checkpoints": len(fault_ids),
            "common_methods": list(methods),
            "base_metrics_supplied": base_metrics_supplied,
        },
        "h3": dict(h3),
    }
