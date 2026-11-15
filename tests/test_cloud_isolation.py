import json
import os
from pathlib import Path

import pytest

import neuroforge.isolation as isolation


def test_tree_hash_is_stable_and_rejects_symlinks(tmp_path: Path) -> None:
    tree = tmp_path / "frozen"
    tree.mkdir()
    (tree / "adapter.bin").write_bytes(b"adapter")
    assert isolation.tree_sha256(tree) == isolation.tree_sha256(tree)
    (tree / "escape").symlink_to(tree / "adapter.bin")
    with pytest.raises(ValueError, match="symlink"):
        isolation.tree_sha256(tree)


def test_freeze_proof_requires_all_bound_artifact_hashes(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "manifest.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("model: frozen\n", encoding="utf-8")
    router = tmp_path / "router.json"
    router.write_text("{}", encoding="utf-8")
    proof = tmp_path / "freeze-proof.json"
    proof.write_text(
        json.dumps(
            {
                "candidate_sha256": isolation.tree_sha256(candidate),
                "config_sha256": isolation.file_sha256(config),
                "router_selection_sha256": isolation.file_sha256(router),
            }
        ),
        encoding="utf-8",
    )
    isolation._freeze_proof(proof, candidate=candidate, config=config, router_selection=router)
    router.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(isolation.IsolationBlocked, match="freeze proof"):
        isolation._freeze_proof(proof, candidate=candidate, config=config, router_selection=router)


def test_unix_gate_command_is_fixed_and_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "job"
    runtime = root / "runtime"
    source = tmp_path / "source"
    source.mkdir()
    runtime.mkdir(parents=True)
    monkeypatch.setattr(isolation, "_trusted_tool", lambda first, *_: first)
    monkeypatch.setattr(isolation, "_trusted_gpu_library_path", lambda: None)
    command = isolation.build_evaluator_command(
        isolation.IsolationPaths(
            root=root,
            runtime=runtime,
            request=runtime / "inputs" / "request.json",
            output=runtime / "output" / "metrics.json",
            source=source,
        )
    )
    rendered = " ".join(command)
    assert command[:2] == ["env", "-i"]
    assert "HF_HUB_OFFLINE=1" in command
    assert "TRANSFORMERS_OFFLINE=1" in command
    assert "TRUST_REMOTE_CODE=false" in command
    assert "--reuid=2100" in command
    assert "--regid=2100" in command
    assert "--bounding-set=-all" in command
    assert "--no-new-privs" in command
    assert "neuroforge.evaluation" in command
    assert "--privileged" not in rendered
    assert "--env-host" not in rendered
    assert "NEUROFORGE_EVALUATOR_ROOT=" in rendered


@pytest.mark.skipif(
    os.environ.get("NEUROFORGE_RUN_UID_BOUNDARY_TESTS") != "1"
    or not os.environ.get("NEUROFORGE_UID_PROBE_ROOT")
    or not os.environ.get("NEUROFORGE_UID_PROBE_SOURCE")
    or not os.environ.get("NEUROFORGE_UID_PROBE_MARKER"),
    reason="opt-in test requires deployment-prepared UID probe paths",
)
def test_deployment_uid_probe_is_explicitly_opt_in() -> None:
    root = Path(os.environ["NEUROFORGE_UID_PROBE_ROOT"])
    runtime = root / "runtime"
    paths = isolation.IsolationPaths(
        root=root,
        runtime=runtime,
        request=runtime / "inputs" / "request.json",
        output=runtime / "output" / "metrics.json",
        source=Path(os.environ["NEUROFORGE_UID_PROBE_SOURCE"]),
    )
    assert isolation.probe_authority(
        paths, sealed_marker=os.environ["NEUROFORGE_UID_PROBE_MARKER"]
    ) == {
        "repair_cannot_read_sealed_marker_or_edit_gate_file": True,
        "gate_can_read_marker_and_source_but_cannot_write_source": True,
    }
