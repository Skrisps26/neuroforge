"""Mocked upstream/unit edges only; no Tiny Aya model integration is implied."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pytest

from neuroforge.artifacts import ArtifactIntegrityError, write_manifest, write_tensor_artifact
from neuroforge.contracts import DataItem, GateDecision, GateOutcome, Metrics, PatchManifest, RunConfig
from neuroforge.modeling import generate_responses
from neuroforge.repairs import RepairUnavailable, _circuit_layers, _expected_qv_mask, _random_layers


def _diagnostics() -> dict[str, object]:
    return {
        "selected_layers": [2, 4, 6, 8],
        "D3": {
            "selected_layers": [2, 4, 6, 8],
            "qv_whole_matrix_mask": ["layers.2.q_proj.weight", "layers.2.v_proj.weight"],
            "random_controls": {
                "101": {
                    "layers": [1, 3, 5, 7],
                    "qv_whole_matrix_mask": ["layers.1.q_proj.weight", "layers.1.v_proj.weight"],
                }
            },
        },
    }


def test_circuit_mask_consumes_actual_diagnostics_schema() -> None:
    diagnostics = _diagnostics()
    assert _circuit_layers(diagnostics) == [2, 4, 6, 8]
    with pytest.raises(RepairUnavailable, match="disagree"):
        _circuit_layers({"selected_layers": [2, 4, 6, 8], "D3": {"selected_layers": [1, 3, 5, 7]}})


def test_random_mask_consumes_recorded_d3_control_only() -> None:
    diagnostics = _diagnostics()
    assert _random_layers(diagnostics, "R0") == [1, 3, 5, 7]
    with pytest.raises(RepairUnavailable, match="seed 102"):
        _random_layers(diagnostics, "R1")


def test_d3_qv_mask_must_match_selected_tensor_capacity() -> None:
    diagnostics = _diagnostics()
    _expected_qv_mask(diagnostics, "M2", ["layers.2.v_proj.weight", "layers.2.q_proj.weight"])
    with pytest.raises(RepairUnavailable, match="exactly match"):
        _expected_qv_mask(diagnostics, "M2", ["layers.2.q_proj.weight"])

def test_generation_tokenizes_the_frozen_utility_prompt_without_reappending_instruction() -> None:
    torch = pytest.importorskip("torch")

    class CapturingTokenizer:
        pad_token_id = 0
        eos_token_id = 6

        def __init__(self) -> None:
            self.tokenized_prompt: str | None = None

        def apply_chat_template(self, messages, **_kwargs):
            self.tokenized_prompt = messages[0]["content"]
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        def decode(self, _tokens, **_kwargs) -> str:
            return "4"

    class GreedyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids=None, attention_mask=None):
            return input_ids

        def generate(self, **_kwargs):
            return torch.tensor([[1, 2, 4]])

    config = RunConfig(
        model_id="CohereLabs/tiny-aya-global",
        model_revision="a" * 40,
        guardian_id="guardian",
        guardian_revision="b" * 40,
        faults=[],
    )
    frozen_prompt = "What is 2 + 2?\nAnswer with the final number only."
    item = DataItem(
        sample_id="utility-1",
        prompt=frozen_prompt,
        category="utility",
        source="test",
        source_revision="test",
        license="test",
    )
    tokenizer = CapturingTokenizer()
    generate_responses(GreedyModel(), tokenizer, [item], config)

    assert tokenizer.tokenized_prompt == frozen_prompt


def test_reload_rejects_changed_scalar_count_mismatch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from neuroforge.repairs import reload_candidate

    artifact = tmp_path / "patch.safetensors"
    artifact_hash = write_tensor_artifact(artifact, {"q.weight": np.ones((1, 1), dtype=np.float32)})
    manifest = PatchManifest(
        base_model="base",
        base_revision="a" * 64,
        fault_id="f1-01",
        method="M1",
        parameters={
            "tensor_dtypes": {"q.weight": "torch.float32"},
            "changed_tensor_count": 1,
            "selected_tensor_names": ["q.weight"],
            "selected_scalar_capacity": 1,
        },
        artifact_path=artifact.name,
        artifact_sha256=artifact_hash,
        changed_parameter_count=0,
        validation_metrics=Metrics(),
        decision=GateDecision(decision=GateOutcome.INCONCLUSIVE),
        rollback_reference="reload immutable checkpoint",
    )
    write_manifest(tmp_path / "manifest.json", manifest)
    model = torch.nn.Module()
    model.add_module("q", torch.nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        model.q.weight.zero_()

    with pytest.raises(ArtifactIntegrityError, match="changed_parameter_count"):
        reload_candidate(model, None, tmp_path, manifest)


def test_corrupt_candidate_artifact_fails_manifest_integrity(tmp_path: Path) -> None:
    artifact = tmp_path / "patch.safetensors"
    artifact_hash = write_tensor_artifact(
        artifact, {"model.layers.0.self_attn.q_proj.weight": np.ones((1, 1), dtype=np.float32)}
    )
    manifest = PatchManifest(
        base_model="base",
        base_revision="a" * 64,
        fault_id="f1-01",
        method="M1",
        parameters={"tensor_dtypes": {"model.layers.0.self_attn.q_proj.weight": "torch.float32"}},
        artifact_path=artifact.name,
        artifact_sha256=artifact_hash,
        changed_parameter_count=1,
        validation_metrics=Metrics(),
        decision=GateDecision(decision=GateOutcome.INCONCLUSIVE),
        rollback_reference="reload immutable checkpoint",
    )
    write_manifest(tmp_path / "manifest.json", manifest)
    artifact.write_bytes(b"corrupt")
    from neuroforge.artifacts import read_manifest

    with pytest.raises(ArtifactIntegrityError):
        read_manifest(tmp_path / "manifest.json")
