"""Unit-only publication tests; every Hub interaction is a local fake."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from neuroforge.artifacts import sha256_file
from neuroforge.publication import (
    PublicationArtifact,
    PublicationError,
    PublicationRequest,
    execute_publication,
    sanitize_public_log,
    stage_publication,
)


def _artifact(path: Path, destination: str) -> PublicationArtifact:
    return PublicationArtifact(source=path, destination=destination, sha256=sha256_file(path))


def _request(tmp_path: Path, *, source: Path | None = None, destination: str = "train/items.json") -> PublicationRequest:
    if source is None:
        source = tmp_path / "train.json"
        if not source.exists():
            source.write_text('[{"prompt_id":"p1"}]', encoding="utf-8")
    return PublicationRequest(
        artifact_root=tmp_path,
        staging_dir=tmp_path / "stage",
        dataset_repo_id="unit/dataset",
        dataset_artifacts=[
            _artifact(source, destination) if source.is_file() else PublicationArtifact(
                source=source, destination=destination, sha256="0" * 64
            )
        ],
        dataset_provenance={"manifest_sha256": "a" * 64, "sources": ["XSTest", "GSM8K"]},
    )


def test_dataset_staging_hashes_payload_and_writes_source_only_card(tmp_path: Path) -> None:
    plan = stage_publication(_request(tmp_path))
    staged = next(item for item in plan.dataset.artifacts if item.path == "train/items.json")
    assert sha256_file(plan.staging_dir / "dataset" / staged.path) == staged.sha256
    assert (plan.staging_dir / "dataset" / staged.path).read_bytes() == (tmp_path / "train.json").read_bytes()


def test_sealed_payload_is_rejected_before_any_file_is_opened(tmp_path: Path) -> None:
    request = _request(tmp_path, source=tmp_path / "sealed" / "not-opened.json", destination="sealed/final.json")

    with pytest.raises(PublicationError):
        stage_publication(request)


def test_escape_and_missing_artifacts_cannot_stage(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.json"
    outside.write_text("[]", encoding="utf-8")
    with pytest.raises(PublicationError, match="escapes root"):
        stage_publication(_request(tmp_path, source=outside))

    with pytest.raises(PublicationError, match="not a regular file"):
        stage_publication(_request(tmp_path, source=tmp_path / "missing.json"))


def test_response_logs_redact_all_completion_fields_and_reject_credentials() -> None:
    sanitized = sanitize_public_log(
        {
            "prompt_id": "p1",
            "metrics": {"utility": 0.5},
            "config": {
                "input_max_tokens": 512,
                "max_new_tokens": 3,
                "token_type_ids": [0, 1],
                "tokenizer_settings": {"padding": "left"},
            },
            "code_edits": ["x"],
            "model_completion": "unsafe original text",
            "response": {"completion": "more original text"},
        }
    )

    assert sanitized["prompt_id"] == "p1"
    assert sanitized["metrics"]["utility"] == 0.5
    assert sanitized["config"]["input_max_tokens"] == 512
    assert sanitized["config"]["max_new_tokens"] == 3
    assert sanitized["config"]["token_type_ids"] == [0, 1]
    assert sanitized["model_completion"] == "[MODEL_COMPLETION_REDACTED]"
    assert sanitized["response"] == "[MODEL_COMPLETION_REDACTED]"
    with pytest.raises(PublicationError, match="credential-bearing"):
        sanitize_public_log({"HF_TOKEN": "never-public"})
    with pytest.raises(PublicationError, match="credential-bearing"):
        sanitize_public_log({"apiKey": "never-public"})


class _FakeHub:
    """Unit-only fake that creates immutable-looking local committed revisions."""

    def __init__(self, root: Path, *, corrupt_after_upload: bool = False) -> None:
        self.root = root
        self.corrupt_after_upload = corrupt_after_upload
        self.revisions: list[str] = []
        self.collection_items: list[SimpleNamespace] = []

    def create_repo(self, repo_id: str, **kwargs: object) -> SimpleNamespace:
        repo_type = kwargs["repo_type"]
        prefix = "datasets/" if repo_type == "dataset" else ""
        return SimpleNamespace(repo_id=repo_id, repo_type=repo_type, url=f"https://huggingface.co/{prefix}{repo_id}")

    def upload_folder(self, *, repo_id: str, folder_path: Path, **kwargs: object) -> SimpleNamespace:
        revision = "a" * 40
        destination = self.root / repo_id.replace("/", "_") / revision
        shutil.copytree(folder_path, destination)
        if self.corrupt_after_upload:
            (destination / "train" / "items.json").write_text("changed", encoding="utf-8")
        repo_type = kwargs["repo_type"]
        prefix = "datasets/" if repo_type == "dataset" else ""
        repo_url = SimpleNamespace(repo_id=repo_id, repo_type=repo_type, url=f"https://huggingface.co/{prefix}{repo_id}")
        return SimpleNamespace(oid=revision, repo_url=repo_url)

    def repo_info(self, repo_id: str, *, revision: str, **kwargs: object) -> SimpleNamespace:
        self.revisions.append(revision)
        return SimpleNamespace(private=False, gated=False)

    def list_repo_files(self, repo_id: str, *, revision: str, **kwargs: object) -> list[str]:
        root = self.root / repo_id.replace("/", "_") / revision
        return [str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()]

    def create_collection(self, title: str, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(slug="unit/neuroforge-data", private=False)

    def add_collection_item(self, collection_slug: str, item_id: str, item_type: str, **kwargs: object) -> SimpleNamespace:
        self.collection_items.append(SimpleNamespace(item_id=item_id, item_type=item_type))
        return SimpleNamespace(slug=collection_slug)

    def get_collection(self, collection_slug: str) -> SimpleNamespace:
        return SimpleNamespace(slug=collection_slug, private=False, items=self.collection_items)


def _install_download_fake(monkeypatch: pytest.MonkeyPatch, hub: _FakeHub) -> None:
    def download(repo_id: str, filename: str, *, revision: str, **kwargs: object) -> str:
        return str(hub.root / repo_id.replace("/", "_") / revision / filename)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))


def test_execution_verifies_returned_commit_revision_and_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = stage_publication(_request(tmp_path))
    hub = _FakeHub(tmp_path / "remote")
    _install_download_fake(monkeypatch, hub)

    report = execute_publication(plan, hub)

    assert report.dataset.commit_oid == "a" * 40
    assert hub.revisions == ["a" * 40]
    assert report.model_collection_status == "incomplete_no_model_payload"


def test_execution_rejects_corruption_after_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = stage_publication(_request(tmp_path))
    hub = _FakeHub(tmp_path / "remote", corrupt_after_upload=True)
    _install_download_fake(monkeypatch, hub)

    with pytest.raises(PublicationError, match="hash mismatch"):
        execute_publication(plan, hub)


def test_model_cannot_be_manufactured_from_base_configuration(tmp_path: Path) -> None:
    request = _request(tmp_path)
    payload = tmp_path / "base-config.json"
    payload.write_text('{"base_model":"only"}', encoding="utf-8")
    request.model_repo_id = "unit/model"
    request.model_artifacts = [_artifact(payload, "config.json")]
    request.model_kind = "accepted_repair"

    with pytest.raises(PublicationError, match="requires explicit payload, manifest, and model_kind"):
        stage_publication(request)
