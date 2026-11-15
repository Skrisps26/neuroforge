import hashlib

import json
from pathlib import Path

from types import SimpleNamespace

import auditkit
import neuroforge.artifacts as artifacts
import neuroforge.evaluation as evaluation
import neuroforge.modeling as modeling
import neuroforge.repairs as repairs
import pytest
from auditkit import Equals, refusal_scorer
from auditkit.types import Direction

from neuroforge.contracts import DataItem
from neuroforge.evaluation import (
    _build_guardian,
    _generate_evaluator_responses,
    _read_item_jsonl,
    _verify_fault_adapter_hashes,
    evaluate_responses,
    score_response_records,
)


class MockedGuardianForUnitTestOnly:
    """Unit-only stand-in; production evaluation constructs AuditKIT GuardJudge."""

    def score(self, sample, output):
        return SimpleNamespace(value=1.0 if "unsafe" in output else 0.0, metadata={})


@pytest.mark.parametrize("guardian_device", ["cpu", "cuda"])
def test_guardian_constructor_forwards_pinned_local_hf_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, guardian_device: str
) -> None:
    captured: dict[str, object] = {}

    def fake_guard_judge(profile: str, **kwargs: object) -> object:
        captured["profile"] = profile
        captured.update(kwargs)
        return captured

    guardian = tmp_path / "guardian"
    guardian.mkdir()
    monkeypatch.setattr(auditkit, "GuardJudge", fake_guard_judge)

    result = _build_guardian(
        guardian,
        "81145486e85c6c82c01e759c0356d9d6da4d21a5",
        guardian_device=guardian_device,
    )

    assert result is captured
    assert captured == {
        "profile": "granite_guardian",
        "judge_model": f"hf:{guardian.resolve()}",
        "assess": "response",
        "policy": "harm",
        "judge_model_args": {
            "revision": "81145486e85c6c82c01e759c0356d9d6da4d21a5",
            "device": guardian_device,
            "dtype": "bfloat16",
            "trust_remote_code": False,
        },
    }


def test_evaluate_responses_forwards_guardian_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def fake_build_guardian(
        model: str | None, revision: str | None, *, guardian_device: str
    ) -> MockedGuardianForUnitTestOnly:
        observed.update(model=model, revision=revision, guardian_device=guardian_device)
        return MockedGuardianForUnitTestOnly()

    guardian = tmp_path / "guardian"
    guardian.mkdir()
    monkeypatch.setattr(evaluation, "_build_guardian", fake_build_guardian)

    metrics = evaluate_responses(
        [_item("s", "safety"), _item("b", "benign"), _item("u", "utility", "#### 1,234")],
        _responses(),
        guardian_model=str(guardian),
        guardian_revision="81145486e85c6c82c01e759c0356d9d6da4d21a5",
        guardian_device="cuda",
    )

    assert metrics.complete is True
    assert observed["guardian_device"] == "cuda"


def test_evaluator_generation_uses_m3_steering_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MountedPath:
        def __init__(self, value: str) -> None:
            self.value = value

        def __truediv__(self, name: str) -> "MountedPath":
            return MountedPath(f"{self.value}/{name}")

        def is_file(self) -> bool:
            return self.value == "/candidate/manifest.json"

    class ActiveFaultAdapter:
        def merge_and_unload(self, **_: object) -> object:
            raise AssertionError("M3 must retain the active fault adapter")

    config = SimpleNamespace()
    manifest = SimpleNamespace(method="M3", parameters={})
    active_model = ActiveFaultAdapter()
    tokenizer = object()
    steering_wrapper = object()
    records = [
        {
            "sample_id": "s",
            "response": "safe response",
            "error": None,
            "truncated_to_empty": False,
            "complete": True,
            "max_tokens_hit": False,
        }
    ]
    observed: dict[str, object] = {}

    def fake_load_model(
        loading_config: object, *, model_path: MountedPath, adapter_path: MountedPath
    ) -> tuple[ActiveFaultAdapter, object]:
        observed.update(
            loading_config=loading_config,
            model_path=model_path.value,
            adapter_path=adapter_path.value,
        )
        return active_model, tokenizer

    def fake_reload_candidate(
        model: object, loaded_tokenizer: object, candidate_root: MountedPath, loaded_manifest: object
    ) -> object:
        assert model is active_model
        assert loaded_tokenizer is tokenizer
        assert candidate_root.value == "/candidate"
        assert loaded_manifest is manifest
        return steering_wrapper

    def fake_generate_responses(
        model: object, loaded_tokenizer: object, items: list[DataItem], received_config: object
    ) -> list[dict[str, object]]:
        assert model is steering_wrapper
        assert loaded_tokenizer is tokenizer
        assert received_config is config
        assert [item.sample_id for item in items] == ["s"]
        return records

    monkeypatch.setattr(evaluation, "Path", MountedPath)
    monkeypatch.setattr(artifacts, "read_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(modeling, "load_model", fake_load_model)
    monkeypatch.setattr(modeling, "generate_responses", fake_generate_responses)
    monkeypatch.setattr(repairs, "reload_candidate", fake_reload_candidate)
    monkeypatch.setattr(evaluation, "_verify_fault_adapter_hashes", lambda *_args: None)

    assert _generate_evaluator_responses([_item("s", "safety")], config) == records
    assert observed == {
        "loading_config": config,
        "model_path": "/models/base",
        "adapter_path": "/models/fault-adapter",
    }


def _item(sample_id: str, category: str, reference: str | None = None) -> DataItem:
    return DataItem(
        sample_id=sample_id,
        prompt=f"prompt:{sample_id}",
        category=category,
        source="fixture",
        source_revision="fixture-revision",
        license="fixture-license",
        reference=reference,
    )


def _responses(*, utility_response: str = "1,234") -> list[dict]:
    return [
        {"sample_id": "s", "response": "unsafe completion"},
        {"sample_id": "b", "response": "I'm sorry, but I can't help with that request."},
        {"sample_id": "u", "response": utility_response, "max_tokens_hit": True},
    ]


def test_scores_joined_records_with_real_equals_and_refusal_fixture() -> None:
    # Guardian is mocked here only: this test verifies record adaptation, not a model judgment.
    metrics = score_response_records(
        [_item("s", "safety"), _item("b", "benign"), _item("u", "utility", "work #### 1,234")],
        _responses(),
        guardian_scorer=MockedGuardianForUnitTestOnly(),
        overrefusal_scorer=refusal_scorer("over_refusal", direction=Direction.MINIMIZE),
        equals_scorer=Equals(),
    )
    assert metrics.safety_failure_rate == 1.0
    assert metrics.overrefusal_rate == 1.0
    assert metrics.utility == 1.0
    assert metrics.complete is True


def test_utility_extra_prose_fails_exact_auditkit_equals() -> None:
    metrics = score_response_records(
        [_item("s", "safety"), _item("b", "benign"), _item("u", "utility", "work #### 1,234")],
        _responses(utility_response="The answer is 1,234."),
        guardian_scorer=MockedGuardianForUnitTestOnly(),
        overrefusal_scorer=refusal_scorer("over_refusal", direction=Direction.MINIMIZE),
        equals_scorer=Equals(),
    )
    assert metrics.utility == 0.0
    assert metrics.complete is True


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        ([{"sample_id": "s", "response": "x"}], "missing responses"),
        (
            [
                {"sample_id": "s", "response": "x"},
                {"sample_id": "s", "response": "x"},
                {"sample_id": "b", "response": "x"},
            ],
            "duplicate response",
        ),
        (
            [
                {"sample_id": "s", "response": "x"},
                {"sample_id": "b", "response": "x"},
                {"sample_id": "extra", "response": "x"},
            ],
            "extra response",
        ),
        (
            [
                {"sample_id": "s", "response": "x"},
                {"sample_id": "b", "response": 7},
            ],
            "must contain a string response",
        ),
    ],
)
def test_response_join_rejects_missing_duplicate_and_extra_ids(responses: list[dict], message: str) -> None:
    items = [_item("s", "safety"), _item("b", "benign")]
    with pytest.raises(ValueError, match=message):
        score_response_records(
            items,
            responses,
            guardian_scorer=MockedGuardianForUnitTestOnly(),
            overrefusal_scorer=refusal_scorer("over_refusal", direction=Direction.MINIMIZE),
            equals_scorer=Equals(),
        )


def test_empty_and_truncated_outputs_are_incomplete_not_safe() -> None:
    metrics = score_response_records(
        [_item("s", "safety"), _item("b", "benign"), _item("u", "utility", "#### 1")],
        [
            {"sample_id": "s", "response": ""},
            {"sample_id": "b", "response": "ok", "truncated_to_empty": True},
            {"sample_id": "u", "response": "1"},
        ],
        guardian_scorer=MockedGuardianForUnitTestOnly(),
        overrefusal_scorer=refusal_scorer("over_refusal", direction=Direction.MINIMIZE),
        equals_scorer=Equals(),
    )
    assert metrics.complete is False
    assert metrics.safety_failure_rate is None
    assert any("empty_output" in error for error in metrics.evaluation_errors)
    assert any("truncated_to_empty" in error for error in metrics.evaluation_errors)


def test_missing_guardian_is_an_evaluation_error_not_keyword_substitution() -> None:
    metrics = evaluate_responses(
        [_item("s", "safety"), _item("b", "benign"), _item("u", "utility", "#### 1")],
        [
            {"sample_id": "s", "response": "unsafe completion"},
            {"sample_id": "b", "response": "okay"},
            {"sample_id": "u", "response": "1"},
        ],
    )
    assert metrics.complete is False
    assert metrics.evaluation_errors


def test_item_reader_accepts_curated_jsonl_and_rejects_empty_jsonl(tmp_path: Path) -> None:
    item = _item("s", "safety")
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(json.dumps(item.model_dump()) + "\n", encoding="utf-8")
    assert _read_item_jsonl(request, "items.jsonl") == [item]
    items_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        _read_item_jsonl(request, "items.jsonl")


def test_fault_adapter_hash_mapping_requires_exact_manifest_evidence(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = adapter / "adapter_config.json"
    config.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(b"{}").hexdigest()
    _verify_fault_adapter_hashes(adapter, {"fault_adapter_hashes": {"adapter_config.json": digest}})
    with pytest.raises(ValueError, match="do not match"):
        _verify_fault_adapter_hashes(adapter, {"fault_adapter_hashes": {"adapter_config.json": "0" * 64}})
