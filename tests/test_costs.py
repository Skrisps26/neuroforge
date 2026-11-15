from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from neuroforge.contracts import CloudPreflight
from neuroforge.costs import BudgetExceeded, BudgetLedger


def ledger(tmp_path: Path) -> BudgetLedger:
    return BudgetLedger(tmp_path / "ledger.jsonl")


def test_operational_and_per_job_limits_require_explicit_approval(tmp_path: Path) -> None:
    budget = ledger(tmp_path)
    budget.reserve("first", Decimal("6.90"), "compute", approve_large_job=True)

    with pytest.raises(BudgetExceeded):
        budget.reserve("reserve", Decimal("0.11"), "compute")
    with pytest.raises(BudgetExceeded):
        budget.reserve("large", Decimal("1.51"), "compute")

    approved = budget.reserve(
        "reserve", Decimal("0.10"), "compute", approve_reserve=True
    )
    assert approved.amount_usd == Decimal("0.10")


def test_absolute_limit_never_yields_to_approvals(tmp_path: Path) -> None:
    budget = ledger(tmp_path)
    with pytest.raises(BudgetExceeded):
        budget.reserve(
            "too-expensive",
            Decimal("10.01"),
            "compute",
            approve_reserve=True,
            approve_large_job=True,
        )
    assert budget.events() == []


@pytest.mark.parametrize("amount", [Decimal("-0.01"), Decimal("NaN"), Decimal("Infinity")])
def test_invalid_amounts_never_enter_ledger(tmp_path: Path, amount: Decimal) -> None:
    budget = ledger(tmp_path)
    with pytest.raises(ValueError):
        budget.reserve("bad", amount, "invalid")
    assert budget.events() == []


def test_historical_and_concurrent_reservations_count_once(tmp_path: Path) -> None:
    budget = ledger(tmp_path)
    budget.record_historical(Decimal("0.63"), "November storage")
    budget.reserve("pod-a", Decimal("1.50"), "job a")
    budget.reserve("pod-b", Decimal("1.50"), "job b")

    assert budget.current_spend() == Decimal("3.63")
    with pytest.raises(ValueError, match="already been recorded"):
        budget.reserve("pod-a", Decimal("0.01"), "retry")


def test_ambiguous_request_stays_reserved_until_reconciled_or_released(tmp_path: Path) -> None:
    budget = ledger(tmp_path)
    budget.reserve("unknown-provider-outcome", Decimal("1.50"), "request")
    assert budget.current_spend() == Decimal("1.50")

    released = budget.release("unknown-provider-outcome", "verified no request")
    assert released.amount_usd == Decimal("1.50")
    assert budget.current_spend() == Decimal("0")
    with pytest.raises(ValueError, match="outstanding reservation"):
        budget.reconcile("unknown-provider-outcome", Decimal("0.10"))


def test_reconciliation_replaces_reservation_without_erasing_observed_cost(tmp_path: Path) -> None:
    budget = ledger(tmp_path)
    budget.reserve("pod", Decimal("1.50"), "job")
    budget.reconcile("pod", Decimal("1.75"), "invoice")

    assert budget.current_spend() == Decimal("1.75")
    with pytest.raises(ValueError, match="outstanding reservation"):
        budget.reconcile("pod", Decimal("0.01"), "second invoice")
    with pytest.raises(ValueError, match="outstanding reservation"):
        budget.release("pod")
    assert budget.current_spend() == Decimal("1.75")


def test_reconciliation_can_record_over_limit_billing_then_blocks_new_requests(tmp_path: Path) -> None:
    budget = BudgetLedger(
        tmp_path / "ledger.jsonl",
        operational_limit=Decimal("1"),
        absolute_limit=Decimal("2"),
        per_job_limit=Decimal("1"),
    )
    budget.reserve("job", Decimal("1"), "job")
    budget.reconcile("job", Decimal("2.01"), "observed bill")

    assert budget.current_spend() == Decimal("2.01")
    with pytest.raises(BudgetExceeded):
        budget.reserve("later", Decimal("0"), "must not proceed")


def test_editor_ledger_blocks_extra_final_submit_above_five_dollars(tmp_path: Path) -> None:
    editor = BudgetLedger(
        tmp_path / "editor.jsonl",
        operational_limit=Decimal("5"),
        absolute_limit=Decimal("5"),
        per_job_limit=Decimal("0.50"),
    )
    editor.reserve("session", Decimal("5.00"), "bounded editor session", approve_large_job=True)

    with pytest.raises(BudgetExceeded):
        editor.reserve("final-submit", Decimal("0.01"), "extra submit turn")


def quoted_preflight(**changes: object) -> CloudPreflight:
    values: dict[str, object] = {
        "purpose": "tiny aya repair",
        "gpu_type": "A40",
        "vram_gb": 48.0,
        "quoted_at": datetime.now(UTC),
        "price_per_hour": Decimal("0.59"),
        "estimated_runtime_seconds": 60.0,
        "hard_runtime_seconds": 3600.0,
        "compute_cost": Decimal("0.59"),
        "storage_cost": Decimal("0.10"),
        "other_cost": Decimal("0.01"),
        "current_spend": Decimal("0"),
        "projected_spend": Decimal("0.70"),
        "output_location": "runs/example",
        "cleanup_plan": "watchdog",
    }
    values.update(changes)
    return CloudPreflight(**values)


def test_preflight_reservation_uses_hard_cap_quote(tmp_path: Path) -> None:
    event = ledger(tmp_path).reserve_preflight("pod", quoted_preflight())
    assert event.amount_usd == Decimal("0.70")


def test_preflight_rejects_underquoted_compute_before_append(tmp_path: Path) -> None:
    budget = ledger(tmp_path)
    preflight = quoted_preflight(compute_cost=Decimal("0.01"))

    with pytest.raises(ValueError, match="hard-runtime cost"):
        budget.reserve_preflight("pod", preflight)
    assert budget.events() == []


def test_preflight_rejects_estimated_runtime_above_hard_cap(tmp_path: Path) -> None:
    budget = ledger(tmp_path)
    preflight = quoted_preflight(
        estimated_runtime_seconds=3601.0,
        hard_runtime_seconds=3600.0,
    )

    with pytest.raises(ValueError, match="estimated runtime"):
        budget.reserve_preflight("pod", preflight)
    assert budget.events() == []


def test_preflight_rejects_stale_current_or_mismatched_projected_spend(
    tmp_path: Path,
) -> None:
    budget = ledger(tmp_path)
    budget.reserve("existing", Decimal("1.00"), "existing job")

    with pytest.raises(ValueError, match="current_spend"):
        budget.reserve_preflight("stale", quoted_preflight())

    consistent_current = quoted_preflight(
        current_spend=Decimal("1.00"),
        projected_spend=Decimal("1.70"),
    )
    with pytest.raises(ValueError, match="projected_spend"):
        budget.reserve_preflight(
            "bad-projection",
            consistent_current.model_copy(
                update={"projected_spend": Decimal("1.69")}
            ),
        )
    assert [event.resource_id for event in budget.events()] == ["existing"]
