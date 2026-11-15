"""Append-only Decimal budget accounting for NeuroForge.

WHY CUSTOM: Python's ``decimal`` and ``fcntl`` provide exact arithmetic and file locking,
but no checked, append-only reservation ledger that composes the study's RunPod and editor
budget rules. No provider SDK is invoked here.
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterator

from .contracts import CloudPreflight, CostEvent


class BudgetExceeded(ValueError):
    """A requested reservation exceeds a non-bypassable study budget."""


class BudgetLedger:
    """Locked JSONL ledger whose current exposure includes unresolved reservations."""

    def __init__(
        self,
        path: Path,
        operational_limit: Decimal = Decimal("7"),
        absolute_limit: Decimal = Decimal("10"),
        per_job_limit: Decimal = Decimal("1.50"),
    ) -> None:
        self.path = Path(path)
        self.operational_limit = self._amount(operational_limit, "operational_limit")
        self.absolute_limit = self._amount(absolute_limit, "absolute_limit")
        self.per_job_limit = self._amount(per_job_limit, "per_job_limit")
        if self.operational_limit > self.absolute_limit:
            raise ValueError("operational_limit must not exceed absolute_limit")

    @staticmethod
    def _amount(value: Decimal, field: str) -> Decimal:
        if not isinstance(value, Decimal):
            raise TypeError(f"{field} must be Decimal")
        if not value.is_finite() or value < 0:
            raise ValueError(f"{field} must be a finite non-negative Decimal")
        return value

    @staticmethod
    def _text(value: str, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        return value

    @contextmanager
    def _locked(self, mode: int) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), mode)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read_events_unlocked(self) -> list[CostEvent]:
        if not self.path.exists():
            return []
        events: list[CostEvent] = []
        with self.path.open("r", encoding="utf-8") as ledger_file:
            for line_number, line in enumerate(ledger_file, start=1):
                if not line.strip():
                    continue
                try:
                    events.append(CostEvent.model_validate_json(line))
                except Exception as exc:  # A damaged financial ledger must not be guessed at.
                    raise ValueError(f"invalid ledger record at line {line_number}") from exc
        return events

    @staticmethod
    def _state(events: list[CostEvent]) -> tuple[Decimal, dict[str, CostEvent]]:
        historical = Decimal("0")
        resources: dict[str, CostEvent] = {}
        event_ids: set[str] = set()
        for event in events:
            if event.event_id in event_ids:
                raise ValueError(f"duplicate event_id in ledger: {event.event_id}")
            event_ids.add(event.event_id)
            if event.event_type == "historical":
                historical += event.amount_usd
                continue
            previous = resources.get(event.resource_id)
            if event.event_type == "reserve":
                if previous is not None:
                    raise ValueError(f"duplicate resource lifecycle: {event.resource_id}")
            elif previous is None or previous.event_type != "reserve":
                raise ValueError(f"invalid resource lifecycle: {event.resource_id}")
            resources[event.resource_id] = event
        current = historical
        for event in resources.values():
            if event.event_type != "release":
                current += event.amount_usd
        return current, resources

    @staticmethod
    def _event(
        event_type: str,
        resource_id: str,
        amount: Decimal,
        purpose: str,
    ) -> CostEvent:
        return CostEvent(
            event_id=uuid.uuid4().hex,
            event_type=event_type,
            resource_id=resource_id,
            amount_usd=amount,
            timestamp=datetime.now(UTC),
            purpose=purpose,
        )

    def _append_unlocked(self, event: CostEvent) -> None:
        payload = json.dumps(
            event.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        with self.path.open("a", encoding="utf-8") as ledger_file:
            ledger_file.write(f"{payload}\n")
            ledger_file.flush()
            os.fsync(ledger_file.fileno())

    def events(self) -> list[CostEvent]:
        """Return validated immutable ledger records in append order."""
        with self._locked(fcntl.LOCK_SH):
            return self._read_events_unlocked()

    def current_spend(self) -> Decimal:
        """Return historic charges plus each resource's latest actual or reservation."""
        with self._locked(fcntl.LOCK_SH):
            current, _ = self._state(self._read_events_unlocked())
            return current

    def _reserve_unlocked(
        self,
        resource_id: str,
        amount: Decimal,
        purpose: str,
        *,
        approve_reserve: bool,
        approve_large_job: bool,
        current: Decimal,
        resources: dict[str, CostEvent],
    ) -> CostEvent:
        """Check and append a reservation while the caller holds the exclusive lock."""
        if resource_id in resources:
            raise ValueError(f"resource_id has already been recorded: {resource_id}")
        if amount > self.per_job_limit and not approve_large_job:
            raise BudgetExceeded(
                f"reservation {amount} exceeds per-job limit {self.per_job_limit}"
            )
        projected = current + amount
        if projected > self.absolute_limit:
            raise BudgetExceeded(
                f"reservation projects {projected}, above absolute limit {self.absolute_limit}"
            )
        if projected > self.operational_limit and not approve_reserve:
            raise BudgetExceeded(
                "reservation enters the protected reserve; explicit approval is required"
            )
        event = self._event("reserve", resource_id, amount, purpose)
        self._append_unlocked(event)
        return event

    @staticmethod
    def _preflight_total(preflight: CloudPreflight) -> Decimal:
        """Validate a quote against its hard runtime before any ledger mutation."""
        estimated_seconds = Decimal(str(preflight.estimated_runtime_seconds))
        hard_seconds = Decimal(str(preflight.hard_runtime_seconds))
        if estimated_seconds > hard_seconds:
            raise ValueError("estimated runtime exceeds the hard runtime cap")
        minimum_compute = preflight.price_per_hour * hard_seconds / Decimal("3600")
        if preflight.compute_cost < minimum_compute:
            raise ValueError("compute quote is below the hard-runtime cost")
        return (
            preflight.compute_cost + preflight.storage_cost + preflight.other_cost
        )

    def reserve(
        self,
        resource_id: str,
        amount: Decimal,
        purpose: str,
        *,
        approve_reserve: bool = False,
        approve_large_job: bool = False,
    ) -> CostEvent:
        """Atomically reserve a worst-case amount before an external request."""
        resource_id = self._text(resource_id, "resource_id")
        purpose = self._text(purpose, "purpose")
        amount = self._amount(amount, "amount")
        with self._locked(fcntl.LOCK_EX):
            current, resources = self._state(self._read_events_unlocked())
            return self._reserve_unlocked(
                resource_id,
                amount,
                purpose,
                approve_reserve=approve_reserve,
                approve_large_job=approve_large_job,
                current=current,
                resources=resources,
            )

    def reconcile(
        self,
        resource_id: str,
        actual_amount: Decimal,
        purpose: str = "billing",
    ) -> CostEvent:
        """Record observed billing exactly once, even when it exceeds its reservation."""
        resource_id = self._text(resource_id, "resource_id")
        purpose = self._text(purpose, "purpose")
        actual_amount = self._amount(actual_amount, "actual_amount")
        with self._locked(fcntl.LOCK_EX):
            _, resources = self._state(self._read_events_unlocked())
            prior = resources.get(resource_id)
            if prior is None:
                raise ValueError("cannot reconcile an unreserved resource")
            if prior.event_type != "reserve":
                raise ValueError("resource is not an outstanding reservation")
            event = self._event("reconcile", resource_id, actual_amount, purpose)
            self._append_unlocked(event)
            return event

    def release(self, resource_id: str, purpose: str = "cleanup") -> CostEvent:
        """Release only an unresolved reservation; observed billing is never erased."""
        resource_id = self._text(resource_id, "resource_id")
        purpose = self._text(purpose, "purpose")
        with self._locked(fcntl.LOCK_EX):
            _, resources = self._state(self._read_events_unlocked())
            prior = resources.get(resource_id)
            if prior is None or prior.event_type != "reserve":
                raise ValueError("only an outstanding reservation can be released")
            event = self._event("release", resource_id, prior.amount_usd, purpose)
            self._append_unlocked(event)
            return event

    def record_historical(self, amount: Decimal, purpose: str) -> CostEvent:
        """Record immutable pre-study billing that must count toward the budget."""
        purpose = self._text(purpose, "purpose")
        amount = self._amount(amount, "amount")
        with self._locked(fcntl.LOCK_EX):
            event = self._event("historical", f"historical:{uuid.uuid4().hex}", amount, purpose)
            self._append_unlocked(event)
            return event

    def reserve_preflight(
        self,
        resource_id: str,
        preflight: CloudPreflight,
        *,
        approve_reserve: bool = False,
        approve_large_job: bool = False,
    ) -> CostEvent:
        """Validate and reserve a hard-cap quote against locked current exposure."""
        if not isinstance(preflight, CloudPreflight):
            raise TypeError("preflight must be CloudPreflight")
        resource_id = self._text(resource_id, "resource_id")
        total = self._amount(self._preflight_total(preflight), "preflight total")
        with self._locked(fcntl.LOCK_EX):
            current, resources = self._state(self._read_events_unlocked())
            if preflight.current_spend != current:
                raise ValueError("preflight current_spend does not match ledger exposure")
            if preflight.projected_spend != current + total:
                raise ValueError("preflight projected_spend does not match quoted total")
            return self._reserve_unlocked(
                resource_id,
                total,
                self._text(preflight.purpose, "preflight purpose"),
                approve_reserve=approve_reserve,
                approve_large_job=approve_large_job,
                current=current,
                resources=resources,
            )
