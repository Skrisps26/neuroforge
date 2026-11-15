from __future__ import annotations

import hashlib

from pathlib import Path

import numpy as np
import pytest

from neuroforge.artifacts import (
    ArtifactIntegrityError,
    promote_artifact,
    read_manifest,
    read_tensor_artifact,
    verify_manifest,
    write_manifest,
    write_tensor_artifact,
)
from neuroforge.contracts import GateDecision, GateOutcome, Metrics, PatchManifest


def _metrics(*, complete: bool = True, errors: list[str] | None = None) -> Metrics:
    return Metrics(
        safety_failure_rate=0.1,
        overrefusal_rate=0.2,
        utility=0.8,
        n_safety=2,
        n_benign=2,
        n_utility=2,
        evaluation_errors=errors or [],
        complete=complete,
    )


def _manifest(
    artifact_sha256: str,
    *,
    decision: GateOutcome = GateOutcome.ACCEPT,
    complete: bool = True,
    include_final_metrics: bool = True,
    final_metrics: Metrics | None = None,
) -> PatchManifest:
    return PatchManifest(
        base_model="CohereLabs/tiny-aya-global",
        base_revision="pinned-revision",
        fault_id="f1-01",
        method="M1",
        artifact_path="patch.safetensors",
        artifact_sha256=artifact_sha256,
        changed_parameter_count=2,
        validation_metrics=_metrics(complete=complete),
        final_metrics=final_metrics if final_metrics is not None else (
            _metrics() if include_final_metrics else None
        ),
        decision=GateDecision(decision=decision),
        rollback_reference="fault-checkpoint-sha256",
    )


def test_safetensors_roundtrip_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "patch.safetensors"
    tensors = {"model.layers.0.self_attn.q_proj.weight": np.array([[1.0, 2.0]], dtype=np.float32)}
    artifact_hash = write_tensor_artifact(path, tensors)

    loaded = read_tensor_artifact(path, artifact_hash)

    assert artifact_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    assert np.array_equal(loaded["model.layers.0.self_attn.q_proj.weight"], tensors[next(iter(tensors))])


def test_corrupt_tensor_artifact_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "patch.safetensors"
    artifact_hash = write_tensor_artifact(path, {"q_proj.weight": np.ones((2, 2), dtype=np.float32)})
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)

    with pytest.raises(ArtifactIntegrityError, match="SHA-256 mismatch"):
        read_tensor_artifact(path, artifact_hash)


def test_manifest_verifies_provenance_and_cannot_overwrite(tmp_path: Path) -> None:
    artifact_hash = write_tensor_artifact(
        tmp_path / "patch.safetensors", {"q_proj.weight": np.ones((1,), dtype=np.float32)}
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = _manifest(artifact_hash)
    write_manifest(manifest_path, manifest)
    loaded = read_manifest(manifest_path)

    verify_manifest(
        loaded,
        expected_base_model="CohereLabs/tiny-aya-global",
        expected_base_revision="pinned-revision",
    )
    with pytest.raises(ArtifactIntegrityError, match="base revision mismatch"):
        verify_manifest(
            loaded,
            expected_base_model="CohereLabs/tiny-aya-global",
            expected_base_revision="other-revision",
        )
    with pytest.raises(ArtifactIntegrityError, match="base model mismatch"):
        verify_manifest(
            loaded,
            expected_base_model="another/base-model",
            expected_base_revision="pinned-revision",
        )
    with pytest.raises(ArtifactIntegrityError, match="overwrite"):
        write_manifest(manifest_path, manifest)


def test_manifest_artifact_cannot_escape_supplied_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact_hash = write_tensor_artifact(
        tmp_path / "outside.safetensors", {"q_proj.weight": np.ones((1,), dtype=np.float32)}
    )
    manifest = _manifest(artifact_hash).model_copy(
        update={"artifact_path": "../outside.safetensors"}
    )

    with pytest.raises(ArtifactIntegrityError, match="escapes root"):
        write_manifest(workspace / "manifest.json", manifest, root=workspace)


def test_rejected_candidate_cannot_promote(tmp_path: Path) -> None:
    artifact_hash = write_tensor_artifact(
        tmp_path / "patch.safetensors", {"q_proj.weight": np.ones((1,), dtype=np.float32)}
    )
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, _manifest(artifact_hash, decision=GateOutcome.REJECT))

    with pytest.raises(ArtifactIntegrityError, match="only ACCEPT"):
        promote_artifact(manifest_path, tmp_path / "accepted")


def test_incomplete_validation_metrics_cannot_promote(tmp_path: Path) -> None:
    artifact_hash = write_tensor_artifact(
        tmp_path / "patch.safetensors", {"q_proj.weight": np.ones((1,), dtype=np.float32)}
    )
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, _manifest(artifact_hash, complete=False))

    with pytest.raises(ArtifactIntegrityError, match="complete validation metrics"):
        promote_artifact(manifest_path, tmp_path / "accepted")


def test_validation_only_accept_cannot_promote(tmp_path: Path) -> None:
    artifact_hash = write_tensor_artifact(
        tmp_path / "patch.safetensors", {"q_proj.weight": np.ones((1,), dtype=np.float32)}
    )
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, _manifest(artifact_hash, include_final_metrics=False))

    with pytest.raises(ArtifactIntegrityError, match="complete final metrics"):
        promote_artifact(manifest_path, tmp_path / "accepted")


@pytest.mark.parametrize(
    "final_metrics",
    [
        _metrics(complete=False),
        _metrics(errors=["sealed evaluator error"]),
    ],
)
def test_incomplete_or_error_final_metrics_cannot_promote(
    tmp_path: Path, final_metrics: Metrics
) -> None:
    artifact_hash = write_tensor_artifact(
        tmp_path / "patch.safetensors", {"q_proj.weight": np.ones((1,), dtype=np.float32)}
    )
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, _manifest(artifact_hash, final_metrics=final_metrics))

    with pytest.raises(ArtifactIntegrityError, match="complete final metrics"):
        promote_artifact(manifest_path, tmp_path / "accepted")


def test_accepted_artifact_promotes_once_by_candidate_hash(tmp_path: Path) -> None:
    artifact_hash = write_tensor_artifact(
        tmp_path / "patch.safetensors", {"q_proj.weight": np.ones((1,), dtype=np.float32)}
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = _manifest(artifact_hash)
    assert manifest.final_metrics is not None
    assert manifest.final_metrics.complete
    write_manifest(manifest_path, manifest)

    promoted = promote_artifact(manifest_path, tmp_path / "accepted")

    assert promoted.name == f"{artifact_hash}.safetensors"
    assert read_tensor_artifact(promoted, artifact_hash)["q_proj.weight"][0] == 1.0
    with pytest.raises(ArtifactIntegrityError, match="already exists"):
        promote_artifact(manifest_path, tmp_path / "accepted")
