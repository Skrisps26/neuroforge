"""Hash-verified, explicit publication staging for NeuroForge artifacts.

This module deliberately separates planning and local staging from Hub mutation.  Callers
must explicitly hand a staged :class:`PublicationPlan` to ``execute_publication``; this
module never obtains credentials or publishes while building a plan.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Protocol

import yaml
from pydantic import Field, field_validator

from .artifacts import ArtifactIntegrityError, read_manifest, sha256_file
from .contracts import ContractModel, FaultConfig, GateOutcome, Metrics, PatchManifest, Sha256Hex, StageResult, config_hash

_CARD_NAME = "README.md"
_DATASET_COLLECTION_TITLE = "NeuroForge: Datasets, Audits and Agent Traces"
_MODEL_COLLECTION_TITLE = "NeuroForge: Models and Patches"
_COMPLETION_PLACEHOLDER = "[MODEL_COMPLETION_REDACTED]"
_COMPLETION_MARKERS = (
    "completion",
    "response",
    "model_output",
    "generated_text",
    "generated_completion",
    "output_text",
    "assistant_message",
    "assistant_content",
)
_CREDENTIAL_COMPONENTS = frozenset({"credential", "password", "secret"})
_TOKEN_CONFIGURATION_COMPONENTS = frozenset(
    {
        "token",
        "tokens",
        "input",
        "output",
        "max",
        "min",
        "new",
        "total",
        "num",
        "count",
        "length",
        "budget",
        "limit",
        "id",
        "ids",
        "eos",
        "bos",
        "pad",
        "generation",
        "type",
        "decoder",
        "encoder",
        "start",
        "forced",
        "special",
        "additional",
    }
)
_ALLOWED_DATASET_PREFIXES = frozenset({"train", "dev", "validation", "provenance"})


class PublicationError(RuntimeError):
    """Raised when a publication request is not safe to stage or verify."""


class HubApi(Protocol):
    """Subset of ``huggingface_hub.HfApi`` used by publication execution."""

    def create_repo(self, repo_id: str, **kwargs: Any) -> Any: ...
    def upload_folder(self, **kwargs: Any) -> Any: ...
    def create_collection(self, title: str, **kwargs: Any) -> Any: ...
    def add_collection_item(self, collection_slug: str, item_id: str, item_type: str, **kwargs: Any) -> Any: ...
    def get_collection(self, collection_slug: str, **kwargs: Any) -> Any: ...
    def repo_info(self, repo_id: str, **kwargs: Any) -> Any: ...
    def list_repo_files(self, repo_id: str, **kwargs: Any) -> list[str]: ...


class PublicationArtifact(ContractModel):
    """One allow-listed regular file and its destination in a public repository."""

    source: Path
    destination: str
    sha256: Sha256Hex

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise ValueError("publication destinations must be relative, normalized paths")
        return path.as_posix()


class PublicationRequest(ContractModel):
    """Strict input for a publication plan; no globbing or inferred payloads."""

    artifact_root: Path
    staging_dir: Path
    dataset_repo_id: str
    dataset_artifacts: list[PublicationArtifact]
    dataset_provenance: dict[str, Any]
    model_repo_id: str | None = None
    model_artifacts: list[PublicationArtifact] = Field(default_factory=list)
    patch_manifest: PublicationArtifact | None = None
    model_kind: Literal["accepted_repair", "fault_adapter"] | None = None
    include_sealed_final: bool = False
    include_optional_extension: bool = False
    final_evaluation_proof: Path | None = None

    @field_validator("dataset_repo_id", "model_repo_id")
    @classmethod
    def validate_repo_id(cls, value: str | None) -> str | None:
        if value is not None and (not value or value.strip() != value):
            raise ValueError("repository identifiers must be non-empty and trimmed")
        return value


class StagedArtifact(ContractModel):
    path: str
    sha256: Sha256Hex
    size_bytes: Annotated[int, Field(ge=1)]


class PublicationPayload(ContractModel):
    repo_id: str
    repo_type: Literal["dataset", "model"]
    artifacts: list[StagedArtifact]


class PublicationPlan(ContractModel):
    """Immutable description of locally staged bytes suitable for a parent-owned upload."""

    schema_version: Literal[1] = 1
    staging_dir: Path
    dataset: PublicationPayload
    model: PublicationPayload | None = None
    model_collection_status: Literal["planned", "incomplete_no_model_payload"]
    report_sha256: Sha256Hex
    stage_result: StageResult


class PublishedRepository(ContractModel):
    repo_id: str
    repo_type: Literal["dataset", "model"]
    repo_url: str
    commit_oid: str
    collection_slug: str
    verified_files: list[StagedArtifact]


class PublicationExecutionReport(ContractModel):
    dataset: PublishedRepository
    model: PublishedRepository | None = None
    model_collection_status: Literal["published", "incomplete_no_model_payload"]


def _has_forbidden_segment(path: Path | PurePosixPath) -> bool:
    return any(part.lower() in {"sealed", "optional", "extension"} for part in path.parts)


def _require_regular_within(path: Path, root: Path) -> Path:
    """Reject symlinks and lexical/root escapes before opening a source file."""
    if root.is_symlink():
        raise PublicationError("artifact root must not be a symlink")
    root_absolute = root.absolute()
    candidate = path if path.is_absolute() else root_absolute / path
    candidate_absolute = candidate.absolute()
    try:
        candidate_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise PublicationError(f"artifact path escapes root: {path}") from exc
    cursor = candidate_absolute
    while cursor != root_absolute:
        if cursor.is_symlink():
            raise PublicationError(f"symlinked publication source is forbidden: {path}")
        cursor = cursor.parent
    if not candidate_absolute.is_file():
        raise PublicationError(f"publication source is not a regular file: {path}")
    resolved_root = root_absolute.resolve(strict=True)
    resolved = candidate_absolute.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise PublicationError(f"resolved publication source escapes root: {path}") from exc
    return resolved


def _authorize_sensitive_inputs(request: PublicationRequest) -> None:
    if not (request.include_sealed_final or request.include_optional_extension):
        return
    proof = request.final_evaluation_proof
    if proof is None:
        raise PublicationError("sealed or optional publication requires final_evaluation_proof")
    proof_path = _require_regular_within(proof, request.artifact_root)
    try:
        proof_value = json.loads(proof_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError("final evaluation proof must be a JSON object") from exc
    if not isinstance(proof_value, dict) or proof_value.get("final_evaluation_complete") is not True:
        raise PublicationError("final evaluation proof does not establish completion")


def _validate_artifact(
    artifact: PublicationArtifact,
    request: PublicationRequest,
    *,
    dataset: bool,
) -> Path:
    source_parts = {part.lower() for part in artifact.source.parts}
    destination_parts = {part.lower() for part in PurePosixPath(artifact.destination).parts}
    parts = source_parts | destination_parts
    if "sealed" in parts and not request.include_sealed_final:
        # This occurs before opening or hashing the file.
        raise PublicationError("sealed artifacts are excluded until final evaluation is explicitly proved")
    if parts & {"optional", "extension"} and not request.include_optional_extension:
        raise PublicationError("optional/extension artifacts are excluded by default")
    source = _require_regular_within(artifact.source, request.artifact_root)
    actual = sha256_file(source)
    if actual != artifact.sha256.lower():
        raise PublicationError(f"allow-listed SHA-256 does not match: {artifact.destination}")
    if dataset:
        parts = PurePosixPath(artifact.destination).parts
        if not parts or parts[0] not in _ALLOWED_DATASET_PREFIXES:
            raise PublicationError("dataset files must be under train/, dev/, validation/, or provenance/")
    return source


def _unique_artifacts(artifacts: Sequence[PublicationArtifact]) -> None:
    destinations = [artifact.destination for artifact in artifacts]
    if len(set(destinations)) != len(destinations):
        raise PublicationError("a publication destination may be staged only once")


def _safe_copy_once(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        source_handle = source.open("rb")
        try:
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except Exception:
            source_handle.close()
            raise
        with source_handle, os.fdopen(descriptor, "wb") as output:
            shutil.copyfileobj(source_handle, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as exc:
        raise PublicationError(f"refusing to overwrite staged artifact: {destination}") from exc


def _write_text_once(destination: Path, text: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise PublicationError(f"refusing to overwrite staged artifact: {destination}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
        output.write(text)
        output.flush()
        os.fsync(output.fileno())


def _field_parts(key: str) -> tuple[str, ...]:
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).lower()
    return tuple(part for part in re.split(r"[^a-z0-9]+", snake_case) if part)


def _is_credential_field(key: str) -> bool:
    """Identify credential-bearing keys without treating token counts as secrets."""
    parts = _field_parts(key)
    joined = "".join(parts)
    if not parts:
        return False
    if set(parts) & _CREDENTIAL_COMPONENTS:
        return True
    if joined in {"apikey", "accesstoken", "authtoken", "bearertoken", "hftoken", "privatekey", "signingkey"}:
        return True
    if {"api", "key"} <= set(parts) or {"access", "key"} <= set(parts) or {"private", "key"} <= set(parts):
        return True
    if "token" in parts:
        return not set(parts).issubset(_TOKEN_CONFIGURATION_COMPONENTS)
    return "authorization" in parts or "cookie" in parts or "session" in parts


def _validate_dataset_provenance(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise PublicationError("dataset provenance keys must be strings")
            lowered = key.lower()
            if _is_credential_field(key):
                raise PublicationError(f"credential-bearing provenance field is forbidden: {key}")
            if any(marker in lowered for marker in ("metric", "model_result", "evaluation")):
                raise PublicationError("dataset source card must not contain model results")
            _validate_dataset_provenance(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_dataset_provenance(nested)


def _dataset_card(provenance: Mapping[str, Any]) -> str:
    # The card intentionally reports source provenance only, never model metrics/results.
    frozen = json.dumps(provenance, sort_keys=True, ensure_ascii=False, indent=2)
    return (
        "---\n"
        "license: other\n"
        "---\n\n"
        "# NeuroForge frozen study data\n\n"
        "## Source licenses\n\n"
        "- XSTest prompts: CC-BY-4.0.\n"
        "- GSM8K: MIT.\n\n"
        "These terms remain distinct; this card does not relicense either source.\n\n"
        "## Frozen provenance metadata\n\n"
        "```json\n"
        f"{frozen}\n"
        "```\n"
    )


def _model_card(kind: str, manifest: PatchManifest | None) -> str:
    if kind == "accepted_repair" and manifest is not None:
        return (
            "---\nlicense: other\n---\n\n# NeuroForge repair payload\n\n"
            f"Verified patch method: `{manifest.method}`. Base: `{manifest.base_model}` at "
            f"`{manifest.base_revision}`. The PatchManifest is the authoritative provenance record.\n"
        )
    return (
        "---\nlicense: other\n---\n\n# NeuroForge fault adapter\n\n"
        "This is an explicitly unaccepted fault artifact, not a repaired model or acceptance claim.\n"
    )


def _complete_metrics(metrics: Metrics | None) -> bool:
    return bool(
        metrics is not None
        and metrics.complete
        and not metrics.evaluation_errors
        and metrics.safety_failure_rate is not None
        and metrics.overrefusal_rate is not None
        and metrics.utility is not None
        and metrics.n_safety > 0
        and metrics.n_benign > 0
        and metrics.n_utility > 0
    )


def _validate_fault_adapter(
    manifest_path: Path, artifacts: Sequence[PublicationArtifact], request: PublicationRequest
) -> None:
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError("fault adapter manifest must be JSON") from exc
    if not isinstance(value, dict):
        raise PublicationError("fault adapter manifest must be a JSON object")
    try:
        FaultConfig.model_validate(value.get("fault"))
    except Exception as exc:
        raise PublicationError("fault adapter manifest lacks a valid fault recipe") from exc
    proof = value.get("reload_proof")
    if not isinstance(proof, dict):
        raise PublicationError("fault adapter lacks recorded reload proof")
    keys = proof.get("canonical_adapter_keys")
    shapes = proof.get("shapes")
    dtypes = proof.get("dtypes")
    if (
        not isinstance(proof.get("adapter_type"), str)
        or not proof["adapter_type"].startswith("PeftModel")
        or proof.get("is_trainable") is not False
        or proof.get("values_match") is not True
        or not isinstance(keys, list)
        or not keys
        or any(not isinstance(key, str) or not key for key in keys)
        or keys != sorted(keys)
        or len(set(keys)) != len(keys)
        or type(proof.get("tensor_count")) is not int
        or proof["tensor_count"] != len(keys)
        or type(proof.get("scalar_count")) is not int
        or proof["scalar_count"] <= 0
        or not isinstance(shapes, dict)
        or not isinstance(dtypes, dict)
        or set(shapes) != set(keys)
        or set(dtypes) != set(keys)
        or any(
            not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            for shape in shapes.values()
        )
        or any(not isinstance(dtype, str) or not dtype for dtype in dtypes.values())
        or proof["scalar_count"] != sum(math.prod(shape) for shape in shapes.values())
    ):
        raise PublicationError("fault adapter reload proof is incomplete or invalid")
    adapter_path = value.get("adapter_path")
    if not isinstance(adapter_path, str):
        raise PublicationError("fault adapter manifest lacks an adapter path")
    adapter_relative = PurePosixPath(adapter_path)
    if not adapter_path or adapter_relative.is_absolute() or ".." in adapter_relative.parts or "." in adapter_relative.parts:
        raise PublicationError("fault adapter manifest has an unsafe adapter path")
    hashes = value.get("adapter_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise PublicationError("fault adapter manifest lacks adapter hashes")
    expected: dict[Path, str] = {}
    for name, digest in hashes.items():
        relative_name = PurePosixPath(name) if isinstance(name, str) else None
        if (
            relative_name is None
            or not name
            or relative_name.is_absolute()
            or ".." in relative_name.parts
            or "." in relative_name.parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            raise PublicationError("fault adapter manifest has invalid adapter hashes")
        source = _require_regular_within(
            manifest_path.parent / adapter_relative / Path(*relative_name.parts), request.artifact_root
        )
        if source in expected:
            raise PublicationError("fault adapter manifest has duplicate adapter files")
        if sha256_file(source) != digest.lower():
            raise PublicationError("fault adapter manifest hash does not match its adapter file")
        expected[source] = digest.lower()
    supplied: dict[Path, str] = {}
    for artifact in artifacts:
        source = _require_regular_within(artifact.source, request.artifact_root)
        if source in supplied:
            raise PublicationError("fault adapter allowlist contains a duplicate source")
        supplied[source] = artifact.sha256.lower()
    if supplied != expected:
        raise PublicationError("fault adapter allowlist does not exactly match its recorded hashes")


def _validate_model(request: PublicationRequest) -> PatchManifest | None:
    if request.model_repo_id is None:
        if request.model_artifacts or request.patch_manifest is not None or request.model_kind is not None:
            raise PublicationError("model settings require model_repo_id")
        return None
    if not request.model_artifacts or request.patch_manifest is None or request.model_kind is None:
        raise PublicationError("model publication requires explicit payload, manifest, and model_kind")
    _unique_artifacts([*request.model_artifacts, request.patch_manifest])
    manifest_path = _validate_artifact(request.patch_manifest, request, dataset=False)
    for artifact in request.model_artifacts:
        _validate_artifact(artifact, request, dataset=False)
    if request.model_kind == "accepted_repair":
        try:
            manifest = read_manifest(manifest_path, root=request.artifact_root)
        except (ArtifactIntegrityError, ValueError) as exc:
            raise PublicationError(f"PatchManifest verification failed: {exc}") from exc
        if manifest.decision.decision != GateOutcome.ACCEPT or not _complete_metrics(manifest.validation_metrics):
            raise PublicationError("accepted repair requires accepted, complete validation metrics")
        if not _complete_metrics(manifest.final_metrics):
            raise PublicationError("accepted repair requires accepted, complete final metrics")
        artifact_path = (manifest_path.parent / manifest.artifact_path).resolve(strict=True)
        if not any(_require_regular_within(item.source, request.artifact_root) == artifact_path for item in request.model_artifacts):
            raise PublicationError("PatchManifest payload is absent from model allowlist")
        return manifest
    _validate_fault_adapter(manifest_path, request.model_artifacts, request)
    return None


def _stage_payload(
    directory: Path,
    artifacts: Sequence[PublicationArtifact],
    request: PublicationRequest,
    *,
    dataset: bool,
) -> list[StagedArtifact]:
    staged: list[StagedArtifact] = []
    for artifact in artifacts:
        source = _validate_artifact(artifact, request, dataset=dataset)
        target = directory / artifact.destination
        _safe_copy_once(source, target)
        actual = sha256_file(target)
        if actual != artifact.sha256.lower():
            raise PublicationError(f"staged artifact changed while copying: {artifact.destination}")
        staged.append(StagedArtifact(path=artifact.destination, sha256=actual, size_bytes=target.stat().st_size))
    return staged


def build_publication_plan(request: PublicationRequest) -> dict[str, Any]:
    """Validate an explicit allowlist without reading/copying sensitive artifacts by default."""
    _authorize_sensitive_inputs(request)
    _validate_dataset_provenance(request.dataset_provenance)
    if not request.dataset_artifacts:
        raise PublicationError("dataset publication requires at least one real data payload")
    _unique_artifacts(request.dataset_artifacts)
    for artifact in request.dataset_artifacts:
        _validate_artifact(artifact, request, dataset=True)
    manifest = _validate_model(request)
    return {
        "dataset_repo_id": request.dataset_repo_id,
        "model_repo_id": request.model_repo_id,
        "model_kind": request.model_kind,
        "accepted_patch_method": None if manifest is None else manifest.method,
        "dataset_payload_count": len(request.dataset_artifacts),
        "model_payload_count": len(request.model_artifacts),
        "model_collection_status": "planned" if request.model_repo_id else "incomplete_no_model_payload",
    }


def stage_publication(request: PublicationRequest) -> PublicationPlan:
    """Copy verified allow-listed files once into a fresh local upload directory."""
    started = time.monotonic()
    plan_summary = build_publication_plan(request)
    stage = request.staging_dir
    if stage.exists():
        raise PublicationError(f"refusing to reuse or overwrite publication staging directory: {stage}")
    stage.mkdir(parents=True, exist_ok=False)
    dataset_dir = stage / "dataset"
    model_dir = stage / "model"
    dataset_artifacts = _stage_payload(dataset_dir, request.dataset_artifacts, request, dataset=True)
    _write_text_once(dataset_dir / _CARD_NAME, _dataset_card(request.dataset_provenance))
    dataset_artifacts.append(
        StagedArtifact(path=_CARD_NAME, sha256=sha256_file(dataset_dir / _CARD_NAME), size_bytes=(dataset_dir / _CARD_NAME).stat().st_size)
    )
    model: PublicationPayload | None = None
    if request.model_repo_id is not None:
        model_artifacts = _stage_payload(model_dir, request.model_artifacts, request, dataset=False)
        manifest = _validate_model(request)
        manifest_artifact = request.patch_manifest  # guarded by _validate_model above
        assert manifest_artifact is not None
        manifest_path = _validate_artifact(manifest_artifact, request, dataset=False)
        staged_manifest = model_dir / manifest_artifact.destination
        _safe_copy_once(manifest_path, staged_manifest)
        manifest_hash = sha256_file(staged_manifest)
        if manifest_hash != manifest_artifact.sha256.lower():
            raise PublicationError("staged patch manifest changed while copying")
        model_artifacts.append(
            StagedArtifact(
                path=manifest_artifact.destination,
                sha256=manifest_hash,
                size_bytes=staged_manifest.stat().st_size,
            )
        )
        _write_text_once(model_dir / _CARD_NAME, _model_card(request.model_kind or "fault_adapter", manifest))
        model_artifacts.append(
            StagedArtifact(path=_CARD_NAME, sha256=sha256_file(model_dir / _CARD_NAME), size_bytes=(model_dir / _CARD_NAME).stat().st_size)
        )
        model = PublicationPayload(repo_id=request.model_repo_id, repo_type="model", artifacts=model_artifacts)
    report = {
        "schema_version": 1,
        "kind": "publication_staging_report",
        "payload_checks": {
            "dataset": [item.model_dump(mode="json") for item in dataset_artifacts],
            "model": [] if model is None else [item.model_dump(mode="json") for item in model.artifacts],
        },
        "plan": plan_summary,
        "network_publication": "not_performed",
    }
    report_path = stage / "publication-report.json"
    report_bytes = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    try:
        descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:  # pragma: no cover - fresh stage makes this defensive
        raise PublicationError("refusing to overwrite publication report") from exc
    with os.fdopen(descriptor, "wb") as output:
        output.write(report_bytes)
        output.flush()
        os.fsync(output.fileno())
    result = StageResult(
        status="success",
        input_artifact_ids=[
            item.sha256
            for item in [
                *request.dataset_artifacts,
                *request.model_artifacts,
                *([] if request.patch_manifest is None else [request.patch_manifest]),
            ]
        ],
        output_artifact_ids=[sha256_file(report_path)],
        config_hash=config_hash(request.model_dump(mode="json")),
        library_versions={"huggingface_hub": "required-at-execution"},
        timing={"wall_seconds": max(time.monotonic() - started, 1e-9)},
    )
    return PublicationPlan(
        staging_dir=stage,
        dataset=PublicationPayload(repo_id=request.dataset_repo_id, repo_type="dataset", artifacts=dataset_artifacts),
        model=model,
        model_collection_status="planned" if model is not None else "incomplete_no_model_payload",
        report_sha256=sha256_file(report_path),
        stage_result=result,
    )


def dry_run_publication(request: PublicationRequest) -> dict[str, Any]:
    """Return validation-only upload intent; unlike staging it performs no writes."""
    return build_publication_plan(request)


def sanitize_public_log(value: Any) -> Any:
    """Remove model completions and reject credentials while retaining structured evidence."""
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise PublicationError("public log keys must be strings")
            lowered = key.lower()
            if _is_credential_field(key):
                raise PublicationError(f"credential-bearing field is forbidden in public log: {key}")
            if any(marker in lowered for marker in _COMPLETION_MARKERS):
                sanitized[key] = _COMPLETION_PLACEHOLDER
            else:
                sanitized[key] = sanitize_public_log(nested)
        return sanitized
    if isinstance(value, list):
        return [sanitize_public_log(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_public_log(item) for item in value]
    return value


def stage_public_log(source: Path, destination: Path, *, root: Path) -> StagedArtifact:
    """Sanitize a JSON response/trace log into a no-overwrite public artifact."""
    input_path = _require_regular_within(source, root)
    try:
        value = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError("public logs must be JSON") from exc
    payload = json.dumps(sanitize_public_log(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    _write_text_once(destination, payload)
    return StagedArtifact(path=destination.name, sha256=sha256_file(destination), size_bytes=destination.stat().st_size)


def _hub_repo_url(value: Any, payload: PublicationPayload) -> str:
    """Require the Hub-returned URL object to identify the repository just created."""
    repo_id = getattr(value, "repo_id", None)
    repo_type = getattr(value, "repo_type", None)
    url = getattr(value, "url", None)
    if repo_id != payload.repo_id or repo_type != payload.repo_type or not isinstance(url, str) or not url:
        raise PublicationError("Hub did not return a URL for the requested repository")
    return url


def _commit_oid(commit: Any, payload: PublicationPayload) -> str:
    value = getattr(commit, "oid", None)
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublicationError("Hub upload did not return a concrete Git commit OID")
    _hub_repo_url(getattr(commit, "repo_url", None), payload)
    return value


def _assert_public_ungated(info: Any) -> None:
    if getattr(info, "private", None) is not False or getattr(info, "gated", None) is not False:
        raise PublicationError("published repository must be public and ungated")


def _validate_remote_artifact(path: Path, artifact: StagedArtifact) -> None:
    if sha256_file(path) != artifact.sha256:
        raise PublicationError(f"downloaded payload hash mismatch: {artifact.path}")
    if artifact.path.endswith(".json"):
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PublicationError(f"downloaded JSON does not parse: {artifact.path}") from exc
        if not isinstance(parsed, (dict, list)):
            raise PublicationError(f"downloaded JSON must be structured: {artifact.path}")
    if artifact.path.endswith(".safetensors"):
        try:
            if artifact.size_bytes > 64 * 1024 * 1024:
                from safetensors import safe_open

                with safe_open(str(path), framework="np") as archive:
                    names = list(archive.keys())
                    if not names:
                        raise PublicationError(f"downloaded safetensors has no tensors: {artifact.path}")
                    view = archive.get_slice(names[0])
                    shape = view.get_shape()
                    if shape:
                        view[tuple(slice(0, 1) for _ in shape)]
                    else:
                        archive.get_tensor(names[0])
                return
            from safetensors.numpy import load_file

            tensors = load_file(str(path))
        except Exception as exc:
            raise PublicationError(f"downloaded safetensors cannot reload: {artifact.path}") from exc
        if not tensors:
            raise PublicationError(f"downloaded safetensors has no tensors: {artifact.path}")


def _verify_repository(api: HubApi, payload: PublicationPayload, commit_oid: str) -> list[StagedArtifact]:
    from huggingface_hub import hf_hub_download

    info = api.repo_info(payload.repo_id, repo_type=payload.repo_type, revision=commit_oid, files_metadata=True)
    _assert_public_ungated(info)
    remote_names = set(api.list_repo_files(payload.repo_id, repo_type=payload.repo_type, revision=commit_oid))
    required_names = {artifact.path for artifact in payload.artifacts}
    if _CARD_NAME not in required_names or not required_names.issubset(remote_names):
        raise PublicationError("published revision is missing expected card or payload files")
    verified: list[StagedArtifact] = []
    downloaded_files: dict[str, Path] = {}
    for artifact in payload.artifacts:
        downloaded = Path(
            hf_hub_download(payload.repo_id, artifact.path, repo_type=payload.repo_type, revision=commit_oid)
        )
        _validate_remote_artifact(downloaded, artifact)
        downloaded_files[artifact.path] = downloaded
        verified.append(artifact)
    card = downloaded_files.get(_CARD_NAME)
    if card is None:
        raise PublicationError("published revision is missing its card")
    try:
        text = card.read_text(encoding="utf-8")
        if not text.startswith("---\n") or "\n---\n" not in text:
            raise ValueError("missing YAML front matter")
        metadata = yaml.safe_load(text.split("\n---\n", 1)[0][4:])
        if not isinstance(metadata, dict) or metadata.get("license") != "other":
            raise ValueError("invalid card metadata")
    except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        raise PublicationError("published card does not parse") from exc
    return verified


def _publish_payload(api: HubApi, payload: PublicationPayload, folder: Path, collection_title: str) -> PublishedRepository:
    repo_url = api.create_repo(payload.repo_id, repo_type=payload.repo_type, private=False, exist_ok=False)
    returned_repo_url = _hub_repo_url(repo_url, payload)
    commit = api.upload_folder(
        repo_id=payload.repo_id,
        folder_path=folder,
        repo_type=payload.repo_type,
        commit_message="Publish hash-verified NeuroForge artifacts",
    )
    oid = _commit_oid(commit, payload)
    verified = _verify_repository(api, payload, oid)
    collection = api.create_collection(collection_title, private=False, exists_ok=False)
    slug = getattr(collection, "slug", None)
    if not isinstance(slug, str) or not slug or getattr(collection, "private", None) is not False:
        raise PublicationError("Hub collection creation did not return a public collection")
    api.add_collection_item(slug, payload.repo_id, payload.repo_type, exists_ok=False)
    observed = api.get_collection(slug)
    items = getattr(observed, "items", None)
    if (
        getattr(observed, "slug", None) != slug
        or getattr(observed, "private", None) is not False
        or not isinstance(items, list)
        or not any(
            getattr(item, "item_id", None) == payload.repo_id and getattr(item, "item_type", None) == payload.repo_type
            for item in items
        )
    ):
        raise PublicationError("created public collection does not contain the published repository")
    return PublishedRepository(
        repo_id=payload.repo_id,
        repo_type=payload.repo_type,
        repo_url=returned_repo_url,
        commit_oid=oid,
        collection_slug=slug,
        verified_files=verified,
    )


def execute_publication(plan: PublicationPlan, api: HubApi) -> PublicationExecutionReport:
    """Perform parent-authorized Hub mutation and verify the exact returned revisions."""
    report = plan.staging_dir / "publication-report.json"
    if not report.is_file() or sha256_file(report) != plan.report_sha256:
        raise PublicationError("staging report is missing or has changed")
    for payload, folder in ((plan.dataset, plan.staging_dir / "dataset"), (plan.model, plan.staging_dir / "model")):
        if payload is None:
            continue
        for artifact in payload.artifacts:
            source = folder / artifact.path
            if not source.is_file() or sha256_file(source) != artifact.sha256:
                raise PublicationError(f"staged payload changed before upload: {artifact.path}")
    dataset = _publish_payload(api, plan.dataset, plan.staging_dir / "dataset", _DATASET_COLLECTION_TITLE)
    model = None
    if plan.model is not None:
        model = _publish_payload(api, plan.model, plan.staging_dir / "model", _MODEL_COLLECTION_TITLE)
    return PublicationExecutionReport(
        dataset=dataset,
        model=model,
        model_collection_status="published" if model is not None else "incomplete_no_model_payload",
    )
