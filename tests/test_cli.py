from __future__ import annotations

import json
from pathlib import Path

import pytest

from typer.testing import CliRunner

from neuroforge.cli import app


def test_doctor_persists_inventory_without_secret_values(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setenv("HF_TOKEN", "must-not-appear-in-an-artifact")

    result = runner.invoke(app, ["doctor", "--run-dir", str(tmp_path / "doctor")])

    assert result.exit_code == 0, result.stdout
    report = json.loads((tmp_path / "doctor" / "doctor" / "report.json").read_text(encoding="utf-8"))
    assert report["auth_configured"]["HF_TOKEN"] is True
    assert "must-not-appear-in-an-artifact" not in (tmp_path / "doctor" / "doctor" / "report.json").read_text(
        encoding="utf-8"
    )




@pytest.mark.parametrize("split", ["sealed", "optional_extension"])
def test_protected_split_is_rejected_before_opening_any_file(tmp_path: Path, split: str) -> None:
    from neuroforge.cli import _load_curated_split

    with pytest.raises(ValueError):
        _load_curated_split(tmp_path / "does-not-exist.json", split, "a" * 64)


def test_run_context_survives_code_commits_but_rejects_config_changes(tmp_path: Path, monkeypatch) -> None:
    from neuroforge.cli import _run_snapshot
    from neuroforge.contracts import load_config

    config = load_config(Path("configs/study.yaml"))
    monkeypatch.setattr("neuroforge.cli._hardware", lambda: {})
    monkeypatch.setattr("neuroforge.cli._git_commit", lambda: "a" * 40)
    original_hash, _ = _run_snapshot(config, tmp_path)
    path = tmp_path / "provenance" / "run-config.json"
    original_bytes = path.read_bytes()
    monkeypatch.setattr("neuroforge.cli._git_commit", lambda: "b" * 40)
    assert _run_snapshot(config, tmp_path)[0] == original_hash
    assert path.read_bytes() == original_bytes
    with pytest.raises(ValueError):
        _run_snapshot(config.model_copy(update={"seed": config.seed + 1}), tmp_path)
