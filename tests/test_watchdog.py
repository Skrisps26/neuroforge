from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts import cloud_watchdog


def allowlist_file(
    tmp_path: Path,
    resources: list[dict[str, object]],
    name: str = "allowlist.json",
) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"resources": resources}), encoding="utf-8")
    return path


def resource(
    resource_id: str,
    resource_kind: str,
    *,
    created_at: datetime | None = None,
    deadline: datetime | None = None,
) -> dict[str, object]:
    created_at = created_at or datetime.now(UTC)
    deadline = deadline or created_at + timedelta(minutes=30)
    return {
        "resource_id": resource_id,
        "resource_kind": resource_kind,
        "created_at": created_at.isoformat(),
        "deadline": deadline.isoformat(),
        "provenance": "MCPcreate",
        "session_id": "neuroforge-session",
    }


def test_cleanup_uses_only_allowlisted_ids_in_pod_then_volume_order(tmp_path: Path) -> None:
    path = allowlist_file(
        tmp_path,
        [resource("volume-1", "volume"), resource("pod-1", "pod")],
    )
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        assert kwargs["shell"] is False
        return subprocess.CompletedProcess(argv, 0, "", "")

    result = cloud_watchdog.cleanup_at_deadline(path, cli_path="/fake/runpodctl", runner=fake_run)

    assert result == {"deleted": ["pod-1", "volume-1"], "failed": [], "planned": []}
    assert calls == [
        ["/fake/runpodctl", "pod", "delete", "pod-1"],
        ["/fake/runpodctl", "network-volume", "delete", "volume-1"],
    ]
    assert not path.exists()


def test_partial_cleanup_retains_failed_resource_in_allowlist(tmp_path: Path) -> None:
    path = allowlist_file(
        tmp_path,
        [resource("pod-1", "pod"), resource("volume-1", "volume")],
    )

    def fake_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, int(argv[-1] == "volume-1"), "", "failed")

    result = cloud_watchdog.cleanup_at_deadline(path, cli_path="fake", runner=fake_run)

    assert result["deleted"] == ["pod-1"]
    assert result["failed"] == ["volume-1"]
    remaining = cloud_watchdog.load_allowlist(path)
    assert [item.resource_id for item in remaining.resources] == ["volume-1"]


def test_volume_created_then_pod_creation_failure_finalizes_volume(tmp_path: Path) -> None:
    path = allowlist_file(tmp_path, [resource("volume-created", "volume")])
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    result = cloud_watchdog.finalize_provisioning_failure(
        path, cli_path="fake-runpodctl", runner=fake_run
    )
    assert result["deleted"] == ["volume-created"]
    assert calls == [["fake-runpodctl", "network-volume", "delete", "volume-created"]]


def test_arming_failure_is_followed_by_recorded_resource_finalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = datetime.now(UTC)
    deadline = created + timedelta(minutes=30)
    path = allowlist_file(
        tmp_path, [resource("pod-created", "pod", created_at=created, deadline=deadline)]
    )
    calls: list[list[str]] = []

    def failed_systemd(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, "", "timer unavailable")

    monkeypatch.setattr(cloud_watchdog.subprocess, "run", failed_systemd)
    with pytest.raises(RuntimeError, match="failed to arm"):
        cloud_watchdog.arm_watchdog(path, deadline, "neuroforge-watchdog")

    def successful_delete(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    cloud_watchdog.finalize_provisioning_failure(path, cli_path="fake", runner=successful_delete)
    assert calls == [["fake", "pod", "delete", "pod-created"]]


def test_invalid_deadline_and_nonallowlisted_id_never_reach_cli(tmp_path: Path) -> None:
    created = datetime.now(UTC)
    invalid = allowlist_file(
        tmp_path,
        [resource("foreign-id", "pod", created_at=created, deadline=created + timedelta(hours=2))],
    )
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    with pytest.raises(ValueError, match="invalid watchdog allowlist"):
        cloud_watchdog.cleanup_at_deadline(invalid, cli_path="fake", runner=fake_run)
    assert calls == []

    only_owned = allowlist_file(tmp_path, [resource("owned-id", "pod")], "owned.json")
    cloud_watchdog.cleanup_at_deadline(only_owned, cli_path="fake", runner=fake_run)
    assert calls == [["fake", "pod", "delete", "owned-id"]]
    assert all("foreign-id" not in call for call in calls)


def test_allowlist_change_after_arming_is_rejected_before_deletion(tmp_path: Path) -> None:
    path = allowlist_file(tmp_path, [resource("pod-1", "pod")])
    digest = cloud_watchdog.allowlist_sha256(path)
    changed = allowlist_file(tmp_path, [resource("pod-2", "pod")])
    assert changed == path
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    with pytest.raises(ValueError, match="changed after watchdog arming"):
        cloud_watchdog.cleanup_at_deadline(
            path, expected_sha256=digest, cli_path="fake", runner=fake_run
        )
    assert calls == []


def test_dry_run_never_invokes_cli_or_removes_allowlist(tmp_path: Path) -> None:
    path = allowlist_file(tmp_path, [resource("pod-1", "pod")])

    def unexpected(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("dry-run must not invoke the CLI")

    result = cloud_watchdog.Watchdog(cli_path="fake", runner=unexpected).cleanup(path, dry_run=True)
    assert result == {"deleted": [], "failed": [], "planned": ["pod-1"]}
    assert path.exists()


def test_canary_proves_timer_mechanism_before_provisioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "canary.marker"

    def firing_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        assert argv[0] == "systemd-run"
        marker.write_text(cloud_watchdog.CANARY_CONTENT + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(cloud_watchdog.subprocess, "run", firing_run)
    result = cloud_watchdog.arm_canary(marker, unit_name="neuroforge-canary-test")
    assert result["unit"] == "neuroforge-canary-test.timer"
    cloud_watchdog.require_canary(marker, timeout_seconds=5)


def test_canary_timeout_blocks_provisioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "never.marker"

    def silent_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(cloud_watchdog.subprocess, "run", silent_run)
    cloud_watchdog.arm_canary(marker, unit_name="neuroforge-canary-test")
    with pytest.raises(RuntimeError, match="did not fire"):
        cloud_watchdog.require_canary(marker, timeout_seconds=0)


def test_verify_timer_matches_recorded_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import datetime

    def fake_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if argv[:3] == ["systemctl", "--user", "is-active"]:
            return subprocess.CompletedProcess(argv, 0, "active\n", "")
        return subprocess.CompletedProcess(
            argv, 0, "NextElapseUSec=Sun 2026-11-15 07:16:11 UTC\n", ""
        )

    monkeypatch.setattr(cloud_watchdog.subprocess, "run", fake_run)
    deadline = datetime(2026, 11, 15, 7, 16, 11, tzinfo=cloud_watchdog.UTC)
    assert cloud_watchdog.verify_timer_armed("neuroforge-job-all.timer", deadline=deadline)[
        "unit"
    ].endswith(".timer")


def test_verify_timer_rejects_inactive_or_mismatched_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime

    def inactive(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, "inactive\n", "")

    monkeypatch.setattr(cloud_watchdog.subprocess, "run", inactive)
    deadline = datetime(2026, 11, 15, 7, 16, 11, tzinfo=cloud_watchdog.UTC)
    with pytest.raises(RuntimeError, match="not active"):
        cloud_watchdog.verify_timer_armed("neuroforge-job-all.timer", deadline=deadline)
