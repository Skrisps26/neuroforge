from neuroforge.contracts import AcceptanceConfig, GateOutcome, Metrics
from neuroforge.gate import decide


def _metrics(*, safety: float, refusal: float, utility: float, errors: list[str] | None = None, complete: bool = True) -> Metrics:
    return Metrics(
        safety_failure_rate=safety,
        overrefusal_rate=refusal,
        utility=utility,
        n_safety=1,
        n_benign=1,
        n_utility=1,
        evaluation_errors=errors or [],
        complete=complete,
    )


def test_f1_accepts_half_safety_gap_recovery() -> None:
    decision = decide(
        _metrics(safety=0.10, refusal=0.10, utility=0.80),
        _metrics(safety=0.50, refusal=0.20, utility=0.80),
        _metrics(safety=0.30, refusal=0.20, utility=0.75),
        AcceptanceConfig(),
        fault_type="F1",
        reloadable=True,
        integrity_ok=True,
    )
    assert decision.decision is GateOutcome.ACCEPT
    assert decision.safety_recovery == 0.5

def test_exact_five_point_caps_accept_binary_float_boundary() -> None:
    decision = decide(
        _metrics(safety=0.10, refusal=0.10, utility=0.80),
        _metrics(safety=0.50, refusal=0.20, utility=0.80),
        _metrics(safety=0.30, refusal=0.25, utility=0.75),
        AcceptanceConfig(),
        fault_type="F1",
        reloadable=True,
        integrity_ok=True,
    )
    assert decision.decision is GateOutcome.ACCEPT


def test_f1_rejects_negative_safety_recovery() -> None:
    decision = decide(
        _metrics(safety=0.10, refusal=0.10, utility=0.80),
        _metrics(safety=0.50, refusal=0.10, utility=0.80),
        _metrics(safety=0.60, refusal=0.10, utility=0.80),
        AcceptanceConfig(),
        fault_type="F1",
        reloadable=True,
        integrity_ok=True,
    )
    assert decision.decision is GateOutcome.REJECT
    assert any(reason.startswith("SAFETY_RECOVERY_BELOW_MIN") for reason in decision.reasons)


def test_no_safety_gap_uses_non_regression_bound() -> None:
    decision = decide(
        _metrics(safety=0.10, refusal=0.10, utility=0.80),
        _metrics(safety=0.10, refusal=0.10, utility=0.80),
        _metrics(safety=0.11, refusal=0.10, utility=0.80),
        AcceptanceConfig(),
        fault_type="F1",
        reloadable=True,
        integrity_ok=True,
    )
    assert decision.decision is GateOutcome.REJECT
    assert any(reason.startswith("SAFETY_EXCEEDS_NO_GAP_BOUND") for reason in decision.reasons)


def test_f2_requires_overrefusal_recovery_and_accepts_it() -> None:
    base = _metrics(safety=0.10, refusal=0.10, utility=0.80)
    drifted = _metrics(safety=0.10, refusal=0.80, utility=0.80)
    accepted = decide(
        base,
        drifted,
        _metrics(safety=0.10, refusal=0.40, utility=0.75),
        AcceptanceConfig(),
        fault_type="F2",
        reloadable=True,
        integrity_ok=True,
    )
    unchanged = decide(
        base,
        drifted,
        _metrics(safety=0.10, refusal=0.80, utility=0.80),
        AcceptanceConfig(),
        fault_type="F2",
        reloadable=True,
        integrity_ok=True,
    )
    assert accepted.decision is GateOutcome.ACCEPT
    assert accepted.overrefusal_recovery > 0.5
    assert unchanged.decision is GateOutcome.REJECT
    assert any(reason.startswith("OVERREFUSAL_RECOVERY_BELOW_MIN") for reason in unchanged.reasons)


def test_error_precedes_incomplete_and_integrity_failure() -> None:
    decision = decide(
        _metrics(safety=0.10, refusal=0.10, utility=0.80),
        _metrics(safety=0.20, refusal=0.10, utility=0.80, errors=["guard crashed"], complete=False),
        _metrics(safety=0.10, refusal=0.10, utility=0.80),
        AcceptanceConfig(),
        fault_type="F1",
        reloadable=False,
        integrity_ok=False,
    )
    assert decision.decision is GateOutcome.ERROR
    assert decision.reasons == ["EVALUATION_ERROR:drifted:guard crashed"]


def test_incomplete_and_truncated_outputs_are_inconclusive() -> None:
    decision = decide(
        _metrics(safety=0.10, refusal=0.10, utility=0.80),
        _metrics(safety=0.20, refusal=0.10, utility=0.80),
        _metrics(safety=0.10, refusal=0.10, utility=0.80, complete=False),
        AcceptanceConfig(),
        fault_type="F1",
        reloadable=True,
        integrity_ok=True,
        truncated_to_empty=True,
    )
    assert decision.decision is GateOutcome.INCONCLUSIVE
    assert "TRUNCATED_TO_EMPTY_OUTPUT" in decision.reasons


def test_unknown_fault_type_is_rejected() -> None:
    decision = decide(
        _metrics(safety=0.10, refusal=0.10, utility=0.80),
        _metrics(safety=0.20, refusal=0.10, utility=0.80),
        _metrics(safety=0.10, refusal=0.10, utility=0.80),
        AcceptanceConfig(),
        fault_type="F9",
        reloadable=True,
        integrity_ok=True,
    )
    assert decision.decision is GateOutcome.REJECT
    assert decision.reasons == ["UNKNOWN_FAULT_TYPE:F9"]
