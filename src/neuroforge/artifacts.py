# WHY CUSTOM: hashlib, safetensors, huggingface_hub, and git provide hashing,
# tensor serialization, publication, and version control respectively; none composes
# immutable local artifacts, manifest provenance, and rollback-safe named patches.
"""Immutable, hash-verified study artifact storage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from .contracts import GateOutcome, Metrics, PatchManifest, canonical_json


class ArtifactIntegrityError(RuntimeError):
    """Raised when an immutable artifact, manifest, or patch is invalid."""


def _resolve_path(path: Path, *, root: Path | None = None) -> Path:
    """Resolve ``path`` and reject a path which escapes an explicitly supplied root."""
    resolved = path.resolve(strict=False)
    if root is None:
        return resolved
    resolved_root = root.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ArtifactIntegrityError(f"artifact path escapes root: {path}") from error
    return resolved


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a regular file using bounded-memory streaming."""
    try:
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as error:
        raise ArtifactIntegrityError(f"artifact is missing: {path}") from error
    return digest.hexdigest()


def verify_artifact(path: Path, expected_sha256: str) -> None:
    """Require an existing artifact to match its recorded SHA-256."""
    if not path.is_file():
        raise ArtifactIntegrityError(f"artifact is missing: {path}")
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ArtifactIntegrityError(
            f"artifact SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}"
        )


def write_json_artifact(path: Path, value: Any, *, root: Path | None = None) -> str:
    """Write canonical JSON once, durably, and return its SHA-256."""
    destination = _resolve_path(path, root=root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode("utf-8")
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ArtifactIntegrityError(f"refusing to overwrite artifact: {destination}") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    _fsync_directory(destination.parent)
    return hashlib.sha256(payload).hexdigest()


def read_json_artifact(
    path: Path, expected_sha256: str | None = None, *, root: Path | None = None
) -> dict[str, Any]:
    """Read a JSON object, optionally verifying its immutable content hash first."""
    source = _resolve_path(path, root=root)
    if expected_sha256 is not None:
        verify_artifact(source, expected_sha256)
    try:
        with source.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"invalid JSON artifact: {source}") from error
    if not isinstance(value, dict):
        raise ArtifactIntegrityError(f"JSON artifact must contain an object: {source}")
    return value


def _write_file_once_atomically(destination: Path, write_temp: Any) -> None:
    """Publish a fully-written temporary file with an atomic no-replace link."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ArtifactIntegrityError(f"refusing to overwrite artifact: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    os.close(descriptor)
    try:
        write_temp(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ArtifactIntegrityError(f"refusing to overwrite artifact: {destination}") from error
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_tensor_mapping(tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not tensors:
        raise ArtifactIntegrityError("tensor artifact must contain at least one tensor")
    validated: dict[str, np.ndarray] = {}
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not name:
            raise ArtifactIntegrityError("tensor names must be nonempty strings")
        if not isinstance(tensor, np.ndarray):
            raise ArtifactIntegrityError(f"tensor {name!r} must be a numpy array")
        if not np.issubdtype(tensor.dtype, np.number):
            raise ArtifactIntegrityError(f"tensor {name!r} must have a numeric dtype")
        if not np.isfinite(tensor).all():
            raise ArtifactIntegrityError(f"tensor {name!r} contains nonfinite values")
        validated[name] = tensor
    return validated


def write_tensor_artifact(
    path: Path, tensors: Mapping[str, np.ndarray], *, root: Path | None = None
) -> str:
    """Write replacement tensors as a no-overwrite safetensors artifact."""
    destination = _resolve_path(path, root=root)
    validated = _validated_tensor_mapping(tensors)
    _write_file_once_atomically(destination, lambda temporary: save_file(validated, str(temporary)))
    return sha256_file(destination)


def read_tensor_artifact(
    path: Path, expected_sha256: str, *, root: Path | None = None
) -> dict[str, np.ndarray]:
    """Verify and load a safetensors artifact into NumPy arrays."""
    source = _resolve_path(path, root=root)
    verify_artifact(source, expected_sha256)
    try:
        return dict(load_file(str(source)))
    except Exception as error:
        raise ArtifactIntegrityError(f"invalid safetensors artifact: {source}") from error


def _manifest_artifact_path(
    manifest_path: Path, manifest: PatchManifest, *, root: Path | None = None
) -> Path:
    artifact = Path(manifest.artifact_path)
    resolved = artifact if artifact.is_absolute() else manifest_path.parent / artifact
    return _resolve_path(resolved, root=root)

def _complete_metrics(metrics: Metrics | None) -> bool:
    if metrics is None:
        return False
    return (
        metrics.complete
        and not metrics.evaluation_errors
        and metrics.safety_failure_rate is not None
        and metrics.overrefusal_rate is not None
        and metrics.utility is not None
        and metrics.n_safety > 0
        and metrics.n_benign > 0
        and metrics.n_utility > 0
    )


def write_manifest(path: Path, manifest: PatchManifest, *, root: Path | None = None) -> str:
    """Persist a manifest only after its referenced artifact verifies exactly."""
    manifest_path = _resolve_path(path, root=root)
    artifact_path = _manifest_artifact_path(manifest_path, manifest, root=root)
    verify_artifact(artifact_path, manifest.artifact_sha256)
    # PatchManifest carries the validation gate result directly; serialization must preserve it.
    serialized = manifest.model_dump(mode="json")
    if PatchManifest.model_validate(serialized).decision != manifest.decision:
        raise ArtifactIntegrityError("manifest validation decision changed during serialization")
    return write_json_artifact(manifest_path, serialized, root=root)


def read_manifest(path: Path, *, root: Path | None = None) -> PatchManifest:
    """Read a schema-v1 manifest and verify the referenced artifact relative to it."""
    manifest_path = _resolve_path(path, root=root)
    raw = read_json_artifact(manifest_path, root=root)
    try:
        manifest = PatchManifest.model_validate(raw)
    except Exception as error:
        raise ArtifactIntegrityError(f"invalid patch manifest: {manifest_path}") from error
    if manifest.schema_version != 1:
        raise ArtifactIntegrityError(f"unsupported manifest schema: {manifest.schema_version}")
    verify_artifact(
        _manifest_artifact_path(manifest_path, manifest, root=root), manifest.artifact_sha256
    )
    return manifest


def verify_manifest(
    manifest: PatchManifest, *, expected_base_model: str, expected_base_revision: str
) -> None:
    """Require a manifest to name the immutable base model and revision expected by its caller."""
    if manifest.base_model != expected_base_model:
        raise ArtifactIntegrityError(
            f"base model mismatch: expected {expected_base_model}, got {manifest.base_model}"
        )
    if manifest.base_revision != expected_base_revision:
        raise ArtifactIntegrityError(
            f"base revision mismatch: expected {expected_base_revision}, got {manifest.base_revision}"
        )


def promote_artifact(manifest_path: Path, accepted_dir: Path) -> Path:
    """Trusted orchestration copies only a complete, verified, accepted candidate once.

    This function grants no editor authority: its sole input is a previously persisted
    manifest and it never accepts a caller-supplied decision or artifact path.
    """
    manifest = read_manifest(manifest_path)
    if manifest.decision.decision is not GateOutcome.ACCEPT:
        raise ArtifactIntegrityError("only ACCEPT candidates may be promoted")
    if not _complete_metrics(manifest.validation_metrics):
        raise ArtifactIntegrityError("accepted candidate lacks complete validation metrics")
    if not _complete_metrics(manifest.final_metrics):
        raise ArtifactIntegrityError("accepted candidate lacks complete final metrics")
    if not manifest.rollback_reference:
        raise ArtifactIntegrityError("accepted candidate lacks a rollback reference")
    source = _manifest_artifact_path(manifest_path.resolve(strict=False), manifest)
    destination_root = accepted_dir.resolve(strict=False)
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"{manifest.artifact_sha256.lower()}{source.suffix}"
    if destination.exists():
        raise ArtifactIntegrityError(f"accepted candidate already exists: {destination}")

    def copy_temp(temporary: Path) -> None:
        with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())

    _write_file_once_atomically(destination, copy_temp)
    verify_artifact(destination, manifest.artifact_sha256)
    return destination


def apply_named_patch(
    state: Mapping[str, np.ndarray], replacement: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Return a fresh state mapping with validated replacement tensors copied in.

    The stored tensors are replacements, never ambiguous deltas.  Unchanged arrays are
    deliberately shared because this helper never mutates them; changed arrays are copied.
    """
    result = dict(state)
    for name, tensor in replacement.items():
        if name not in state:
            raise ArtifactIntegrityError(f"patch names unknown tensor: {name}")
        original = state[name]
        if not isinstance(original, np.ndarray) or not isinstance(tensor, np.ndarray):
            raise ArtifactIntegrityError(f"patch tensor {name!r} must be a numpy array")
        if original.shape != tensor.shape:
            raise ArtifactIntegrityError(f"patch tensor shape mismatch: {name}")
        if original.dtype != tensor.dtype:
            raise ArtifactIntegrityError(f"patch tensor dtype mismatch: {name}")
        if not np.issubdtype(tensor.dtype, np.number) or not np.isfinite(tensor).all():
            raise ArtifactIntegrityError(f"patch tensor is nonfinite or nonnumeric: {name}")
    for name, tensor in replacement.items():
        result[name] = tensor.copy()
    return result


def immutable_snapshot(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Deep-copy a NumPy state for rollback proof; this is not Tiny Aya integration evidence."""
    return {name: value.copy() for name, value in state.items()}


def restore_snapshot(
    state: Mapping[str, np.ndarray], snapshot: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Return an exact copied restoration after rejecting incompatible state/snapshot pairs."""
    if set(state) != set(snapshot):
        raise ArtifactIntegrityError("snapshot keys do not match state keys")
    for name, current in state.items():
        saved = snapshot[name]
        if not isinstance(current, np.ndarray) or not isinstance(saved, np.ndarray):
            raise ArtifactIntegrityError(f"snapshot tensor {name!r} must be a numpy array")
        if current.shape != saved.shape or current.dtype != saved.dtype:
            raise ArtifactIntegrityError(f"snapshot tensor mismatch: {name}")
    return {name: snapshot[name].copy() for name in state}
