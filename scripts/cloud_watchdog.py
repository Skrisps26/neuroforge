#!/usr/bin/env python3
"""Independently scheduled cleanup for only conversation-created RunPod resources.

WHY CUSTOM: Pydantic validates durable allowlists and systemd schedules timers, but neither
provides this project's provenance-bound pod-before-volume cleanup policy. The script never
lists resources, reads credentials, or invokes a shell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class WatchdogResource(BaseModel):
    """One resource that this conversation successfully created through MCP."""

    model_config = ConfigDict(extra="forbid")

    resource_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    resource_kind: Literal["pod", "volume"]
    created_at: datetime
    deadline: datetime
    provenance: Literal["MCPcreate"]
    session_id: str = Field(min_length=1, max_length=128)

    @field_validator("created_at", "deadline")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("timestamps must be UTC")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_deadline(self) -> "WatchdogResource":
        if self.deadline < self.created_at:
            raise ValueError("deadline precedes creation")
        if self.deadline - self.created_at > timedelta(hours=1):
            raise ValueError("deadline exceeds the one-hour maximum")
        return self


class WatchdogAllowlist(BaseModel):
    """Strict, finite authorization set for a single cleanup timer."""

    model_config = ConfigDict(extra="forbid")

    resources: list[WatchdogResource] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_resources(self) -> "WatchdogAllowlist":
        ids = [resource.resource_id for resource in self.resources]
        if len(ids) != len(set(ids)):
            raise ValueError("allowlist contains duplicate resource IDs")
        return self


class Watchdog:
    """Run an allowlisted cleanup with an injectable executable for isolated tests."""

    def __init__(
        self,
        cli_path: str | Path = "runpodctl",
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.cli_path = str(cli_path)
        self._runner = runner

    def cleanup(
        self,
        allowlist_path: Path,
        *,
        dry_run: bool = False,
        expected_sha256: str | None = None,
    ) -> dict[str, list[str]]:
        return cleanup_at_deadline(
            allowlist_path,
            dry_run=dry_run,
            expected_sha256=expected_sha256,
            cli_path=self.cli_path,
            runner=self._runner,
        )


def _allowlist_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read allowlist: {path}") from exc


def allowlist_sha256(path: Path) -> str:
    """Return the exact file digest bound into the timer command."""
    return hashlib.sha256(_allowlist_bytes(path)).hexdigest()


def load_allowlist(path: Path, *, expected_sha256: str | None = None) -> WatchdogAllowlist:
    """Read a strict allowlist and reject timer-time edits before any deletion."""
    raw = _allowlist_bytes(Path(path))
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("allowlist changed after watchdog arming")
    try:
        return WatchdogAllowlist.model_validate_json(raw)
    except Exception as exc:
        raise ValueError("invalid watchdog allowlist") from exc


def _write_allowlist(path: Path, allowlist: WatchdogAllowlist) -> None:
    path = Path(path)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            allowlist.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def cleanup_at_deadline(
    allowlist_path: Path,
    *,
    dry_run: bool = False,
    expected_sha256: str | None = None,
    cli_path: str | Path = "runpodctl",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, list[str]]:
    """Delete recorded pods before volumes and leave failed deletions allowlisted."""
    path = Path(allowlist_path)
    allowlist = load_allowlist(path, expected_sha256=expected_sha256)
    ordered = sorted(
        allowlist.resources,
        key=lambda resource: (0 if resource.resource_kind == "pod" else 1, resource.resource_id),
    )
    deleted: list[str] = []
    failed: list[str] = []
    planned: list[str] = []
    for resource in ordered:
        argv = [
            str(cli_path),
            "pod" if resource.resource_kind == "pod" else "network-volume",
            "delete",
            resource.resource_id,
        ]
        if dry_run:
            planned.append(resource.resource_id)
            continue
        completed = runner(argv, shell=False, check=False, text=True, capture_output=True)
        if completed.returncode == 0:
            deleted.append(resource.resource_id)
        else:
            failed.append(resource.resource_id)
    if deleted and not dry_run:
        remaining = [
            resource
            for resource in allowlist.resources
            if resource.resource_id not in set(deleted)
        ]
        if remaining:
            _write_allowlist(path, WatchdogAllowlist(resources=remaining))
        else:
            path.unlink()
    return {"deleted": deleted, "failed": failed, "planned": planned}


def finalize_provisioning_failure(
    allowlist_path: Path,
    *,
    cli_path: str | Path = "runpodctl",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, list[str]]:
    """Finalizer for pod-create or watchdog-arm failures after resource creation."""
    return cleanup_at_deadline(allowlist_path, cli_path=cli_path, runner=runner)


def arm_watchdog(
    allowlist_path: Path, deadline: datetime, unit_name: str, *, dry_run: bool = False
) -> dict[str, str]:
    """Schedule an independent user-systemd timer and verify it was registered."""
    if not unit_name or any(character.isspace() for character in unit_name):
        raise ValueError("unit_name must be a non-empty systemd unit stem")
    if deadline.tzinfo is None or deadline.utcoffset() != timedelta(0):
        raise ValueError("deadline must be UTC")
    deadline = deadline.astimezone(UTC)
    allowlist = load_allowlist(Path(allowlist_path))
    if any(resource.deadline != deadline for resource in allowlist.resources):
        raise ValueError("arming deadline does not exactly match the immutable allowlist")
    digest = allowlist_sha256(Path(allowlist_path))
    timer_unit = f"{unit_name}.timer"
    command = [
        "systemd-run",
        "--user",
        f"--unit={unit_name}",
        f"--on-calendar={deadline.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        sys.executable,
        str(Path(__file__).resolve()),
        "--allowlist",
        str(Path(allowlist_path).resolve()),
        "--expected-sha256",
        digest,
    ]
    if dry_run:
        command.append("--dry-run")
    scheduled = subprocess.run(command, shell=False, check=False, text=True, capture_output=True)
    if scheduled.returncode != 0:
        raise RuntimeError(f"failed to arm watchdog: {scheduled.stderr.strip()}")
    registered = subprocess.run(
        ["systemctl", "--user", "is-active", timer_unit],
        shell=False,
        check=False,
        text=True,
        capture_output=True,
    )
    if registered.returncode != 0:
        raise RuntimeError(f"watchdog timer was not registered: {registered.stderr.strip()}")
    return {"unit": timer_unit, "deadline": deadline.isoformat(), "allowlist_sha256": digest}


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected-sha256")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    result = cleanup_at_deadline(
        arguments.allowlist,
        dry_run=arguments.dry_run,
        expected_sha256=arguments.expected_sha256,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
