from __future__ import annotations

import math
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from neuroforge.contracts import DataItem
from neuroforge.diagnostics import (
    _D2ParityFailure,
    _d2_accounting,
    _d2_patching,
    _minimum_rank_with_safetune,
    _new_d2_accounting,
    _next_token_kl,
    _validate_d2_parity,
    compute_qv_drift,
    minimum_restoring_rank,
)


def test_d1_toy_torch_svd_metrics_are_not_tiny_aya_evidence() -> None:
    # Toy tensor/SVD proof only; this does not load or establish Tiny Aya integration.
    base = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros((2, 2)),
        "model.layers.0.self_attn.v_proj.weight": torch.zeros((2, 2)),
        "model.layers.0.mlp.down_proj.weight": torch.ones((2, 2)),
    }
    drifted = {
        "model.layers.0.self_attn.q_proj.weight": torch.diag(torch.tensor([3.0, 4.0])),
        "model.layers.0.self_attn.v_proj.weight": torch.zeros((2, 2)),
        "model.layers.0.mlp.down_proj.weight": torch.full((2, 2), 99.0),
    }

    result = compute_qv_drift(base, drifted)

    assert result.changed_matrices == 1
    assert result.stable_rank == pytest.approx(25.0 / 16.0)
    assert result.top1_energy == pytest.approx(16.0 / 25.0)
    assert result.top8_energy == pytest.approx(1.0)
    assert result.relative_drift_norm is None
    assert result.changed_parameter_names == ("model.layers.0.self_attn.q_proj.weight",)


def test_d1_known_zero_delta_is_undefined_not_zero_concentration() -> None:
    state = {
        "model.layers.0.self_attn.q_proj.weight": torch.eye(2),
        "model.layers.0.self_attn.v_proj.weight": torch.eye(2),
    }

    result = compute_qv_drift(state, {name: value.clone() for name, value in state.items()})

    assert result.zero_whole_model_drift
    assert result.stable_rank is None
    assert result.top1_energy is None
    assert result.top8_energy is None
    assert result.relative_drift_norm == 0.0


def test_minimum_rank_is_bounded_bisection_not_an_exhaustive_loop() -> None:
    calls: list[int] = []
    scores = {0: 0.0, 2: 2.0, 3: 4.0, 4: 5.0, 8: 10.0}

    result = minimum_restoring_rank(
        lambda rank: calls.append(rank) or scores[rank],
        base_proxy=10.0,
        adapter_rank=8,
        max_bisections=4,
        max_evaluator_calls=6,
    )

    assert result.minimum_rank == 4
    assert result.status == "certified_under_measured_monotonicity"
    assert calls == [0, 8, 4, 2, 3]
    assert result.evaluator_calls == len(calls)
    assert set(calls) != set(range(9))


def test_minimum_rank_reports_nonmonotone_upper_bound() -> None:
    scores = {0: 0.0, 4: 11.0, 8: 10.0}

    result = minimum_restoring_rank(
        lambda rank: scores[rank],
        base_proxy=10.0,
        adapter_rank=8,
        max_evaluator_calls=4,
    )

    assert result.status == "undefined_nonmonotone"
    assert result.minimum_rank is None
    assert result.measured_passing_upper_bound == 4
    assert "nonmonotone" in result.warnings[0]


def test_minimum_rank_respects_evaluator_budget() -> None:
    calls: list[int] = []

    result = minimum_restoring_rank(
        lambda rank: calls.append(rank) or float(rank),
        base_proxy=16.0,
        adapter_rank=16,
        max_bisections=4,
        max_evaluator_calls=3,
    )

    assert len(calls) == 3
    assert result.evaluator_calls == 3
    assert result.status == "undefined_missing_bracket"
    assert any("budget" in warning for warning in result.warnings)


def test_next_token_parity_rejects_kl_over_preregistered_gate() -> None:
    hf_logits = torch.zeros((1, 3, 4))
    identical = _next_token_kl(hf_logits, hf_logits.clone())
    mismatched = _next_token_kl(hf_logits, torch.tensor([[[0.0, 0.0, 0.0, 0.0]] * 2 + [[20.0, 0.0, 0.0, 0.0]]]))

    assert identical == pytest.approx(0.0)
    assert math.isfinite(mismatched)
    assert mismatched >= 1e-4


def test_drifted_checkpoint_parity_rejects_the_same_gate_as_base() -> None:
    accounting = _new_d2_accounting("initial")
    hf_logits = torch.zeros((1, 3, 4))
    mismatched = torch.tensor([[[0.0, 0.0, 0.0, 0.0]] * 2 + [[20.0, 0.0, 0.0, 0.0]]])

    with pytest.raises(_D2ParityFailure, match="drifted") as failure:
        _validate_d2_parity("drifted", "dev-drift", hf_logits, mismatched, accounting)

    assert failure.value.checkpoint == "drifted"
    assert failure.value.kl >= 1e-4


def test_bf16_drift_parity_failure_retries_float32_and_records_all_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model:
        def __init__(self) -> None:
            self.parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
            self.moves: list[dict[str, object]] = []

        def parameters(self):
            yield self.parameter

        def to(self, **kwargs):
            self.moves.append(kwargs)
            return self

    def item(sample_id: str, category: str) -> DataItem:
        return DataItem(
            sample_id=sample_id,
            prompt=sample_id,
            category=category,
            source="unit",
            source_revision="unit",
            license="unit",
        )

    base, drifted = Model(), Model()
    safety, benign = [item("safety", "safety")], [item("benign", "benign")]
    calls: list[dict[str, object]] = []

    def fake_attempt(*args, **kwargs):
        calls.append(kwargs)
        if kwargs["mode"] == "initial":
            initial = _new_d2_accounting("initial")
            initial["hf_base_parity_forwards"] = 1
            initial["transformerlens_cache_forwards"] = 1
            raise _D2ParityFailure("drifted", "safety", 1e-2, initial)
        retry = _new_d2_accounting("float32_retry")
        retry.update(
            hf_base_parity_forwards=2,
            hf_drifted_parity_forwards=2,
            transformerlens_cache_forwards=2,
            transformerlens_drift_forwards=2,
            generic_activation_patch_forwards=12,
        )
        prompt_results = [
            {
                "sample_id": entry.sample_id,
                "category": entry.category,
                "last_position": 1,
                "base_hf_transformerlens_kl": 0.0,
                "drifted_hf_transformerlens_kl": 0.0,
                "recoveries": {str(layer): 1.0 for layer in (0, 6, 12, 18, 24, 30)},
            }
            for entry in args[3]
        ]
        return prompt_results, object(), retry

    monkeypatch.setattr("neuroforge.diagnostics._d2_attempt", fake_attempt)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "neuroforge.diagnostics.refusal_token_pair",
        lambda *_: ((" I", 1), (" Sure", 2)),
    )

    result, _ = _d2_patching(base, drifted, object(), safety, benign, object())

    assert result["parity_retry"]["performed"] is True
    assert result["parity_retry"]["initial_bf16_failure"]["checkpoint"] == "drifted"
    assert calls[1]["serial_bridges"] is True
    assert result["forward_accounting"]["actual_forward_calls"] == 22
    assert base.moves == [{"device": "cpu", "dtype": torch.float32}]
    assert drifted.moves == [{"device": "cpu", "dtype": torch.float32}]


def test_d2_full_success_forward_count_includes_both_checkpoint_parity_calls() -> None:
    attempt = _new_d2_accounting("initial")
    attempt.update(
        hf_base_parity_forwards=16,
        hf_drifted_parity_forwards=16,
        transformerlens_cache_forwards=16,
        transformerlens_drift_forwards=16,
        generic_activation_patch_forwards=96,
    )

    accounting = _d2_accounting([attempt])

    assert accounting["actual_forward_calls"] == 160
    assert accounting["prompt_equivalent_forwards"] == 160


def test_minimum_rank_safetune_scope_is_limited_to_q_and_v_projections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.restored = False

    applied: list[dict[str, object]] = []

    def restore(candidate, reference, **kwargs):
        del reference
        candidate.restored = True
        applied.append(kwargs)

    safetune = types.ModuleType("safetune")
    interventions = types.ModuleType("safetune.interventions")
    recover = types.ModuleType("safetune.interventions.recover")
    restore_module = types.ModuleType("safetune.interventions.recover.safety_vector_restore")
    restore_module.apply_safety_vector_restore = restore
    monkeypatch.setitem(sys.modules, "safetune", safetune)
    monkeypatch.setitem(sys.modules, "safetune.interventions", interventions)
    monkeypatch.setitem(sys.modules, "safetune.interventions.recover", recover)
    monkeypatch.setitem(sys.modules, "safetune.interventions.recover.safety_vector_restore", restore_module)
    monkeypatch.setattr(
        "neuroforge.diagnostics._mean_proxy",
        lambda model, *_: 1.0 if model.restored or model is base else 0.0,
    )

    base, drifted = Model(), Model()
    _minimum_rank_with_safetune(base, drifted, object(), ["prompt"], ((" I", 1), (" Sure", 2)), object(), 1)

    assert applied
    assert all(call["target_modules"] == ["q_proj", "v_proj"] for call in applied)


def test_transformerlens_patch_setter_changes_only_requested_full_cache_position() -> None:
    circuitkit = pytest.importorskip("circuitkit")
    transformer_lens = pytest.importorskip("transformer_lens")
    from transformer_lens.patching import layer_pos_patch_setter

    del circuitkit, transformer_lens
    corrupted = torch.zeros((1, 4, 2))
    full_clean_cache_axis = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)

    patched = layer_pos_patch_setter(corrupted, [6, 3], full_clean_cache_axis)

    assert torch.equal(patched[:, :3, :], torch.zeros((1, 3, 2)))
    assert torch.equal(patched[:, 3, :], full_clean_cache_axis[:, 3, :])
