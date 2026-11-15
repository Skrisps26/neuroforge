"""Per-fault stage isolation: one fault's record must never satisfy another's resume."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "scripts")

import run_study
from neuroforge.artifacts import write_json_artifact


def _args(phase, fault_id):
    return argparse.Namespace(
        config=Path("configs/study.yaml"),
        run_dir=Path("runs"),
        data_manifest=Path("runs/core-curation/data/manifest.json"),
        phase=phase,
        deadline=None,
        hard_budget=None,
        fault_id=fault_id,
    )


def test_stage_key_scopes_only_per_fault_phases():
    for phase in ("population", "diagnostics", "candidates", "sealed"):
        assert run_study._stage_key(_args(phase, "f1-02")) == "f1-02"
        assert run_study._stage_key(_args(phase, None)) is None
    for phase in ("baseline", "first-proof", "route"):
        assert run_study._stage_key(_args(phase, "f1-02")) is None
        assert run_study._stage_key(_args(phase, None)) is None


def test_fault_stage_files_do_not_collide(tmp_path):
    run_dir = tmp_path / "runs"
    (run_dir / "study-execution").mkdir(parents=True)
    first = run_study._stage_path(run_dir, "population", "f1-01")
    second = run_study._stage_path(run_dir, "population", "f1-02")
    assert first != second
    write_json_artifact(
        first,
        {
            "phase": "population",
            "fault_id": "f1-01",
            "result": {
                "status": "success",
                "config_hash": "d",
                "input_artifact_ids": [],
                "output_artifact_ids": [],
                "library_versions": {},
                "timing": {},
                "warnings": [],
                "errors": [],
            },
        },
        root=run_dir,
    )
    assert run_study._existing_stage(run_dir, "population", "d", fault_id="f1-01") is not None
    assert run_study._existing_stage(run_dir, "population", "d", fault_id="f1-02") is None
    assert run_study._existing_stage(run_dir, "population", "d") is None
