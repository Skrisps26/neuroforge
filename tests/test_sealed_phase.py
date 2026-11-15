"""Sealed-phase pure helpers: request shape and output validation."""

import sys
from pathlib import Path

sys.path.insert(0, "scripts")

import run_study


def _config():
    from neuroforge.cli import _load_config

    return _load_config(Path("configs/study.yaml"))


def test_sealed_request_doc_freezes_config():
    from neuroforge.cli import _run_snapshot

    config = _config()
    digest, _ = _run_snapshot(config, Path("runs"))
    for target in ("base", "candidate"):
        doc = run_study._sealed_request_doc(config, digest, target)
        assert doc["mode"] == "generate"
        assert doc["target"] == target
        assert doc["config_hashes"] == {"run_config": digest}
        assert doc["items_path"] == "sealed/sealed.jsonl"
        assert doc["responses_path"] is None
        assert set(doc) == {
            "mode",
            "target",
            "config_hashes",
            "run_config",
            "items_path",
            "responses_path",
        }


def test_sealed_request_doc_rejects_target():
    from neuroforge.cli import _run_snapshot

    config = _config()
    digest, _ = _run_snapshot(config, Path("runs"))
    try:
        run_study._sealed_request_doc(config, digest, "drifted")
    except ValueError:
        return
    raise AssertionError("bad sealed target was accepted")


def test_sealed_methods_lists_only_measured(tmp_path):
    run_dir = tmp_path / "runs"
    (run_dir / "repairs" / "f1-01" / "B0").mkdir(parents=True)
    (run_dir / "repairs" / "f1-01" / "B0" / "validation-record.json").write_text("{}")
    (run_dir / "repairs" / "f1-01" / "M1").mkdir(parents=True)
    assert run_study._sealed_methods(run_dir, "f1-01") == ["B0"]
    (run_dir / "repairs" / "f1-01" / "M1" / "validation-record.json").write_text("{}")
    assert run_study._sealed_methods(run_dir, "f1-01") == ["B0", "M1"]
