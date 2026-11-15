"""Development-only checkpoint diagnostics.

WHY CUSTOM: PyTorch supplies the required SVD primitive, TransformerLens supplies
activation patching, and CircuitKIT supplies attribution and random controls.  This
module only composes those fixed APIs with the preregistered prompt/budget/provenance
rules; it does not implement a substitute circuit, patch hook, or SVD algorithm.
"""
from __future__ import annotations

import copy
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .artifacts import write_json_artifact
from .contracts import DataItem, DiagnosticFeatures, FaultConfig, RunConfig, StageResult, config_hash

_BOUNDARY_LAYERS = (0, 6, 12, 18, 24, 30)
_QV_WEIGHT = re.compile(r"(?:^|\.)(?:q_proj|v_proj)\.weight$")
_HEAD_SCORE = re.compile(r"^(?:a(\d+)\.h(\d+)|A(\d+)\.(\d+))$")

_D2_FORWARD_KEYS = (
    "hf_base_parity_forwards",
    "hf_drifted_parity_forwards",
    "transformerlens_cache_forwards",
    "transformerlens_drift_forwards",
    "generic_activation_patch_forwards",
)


class _D2ParityFailure(RuntimeError):
    """A parity-gate failure that retains the forwards already executed."""

    def __init__(
        self,
        checkpoint: str,
        sample_id: str,
        kl: float,
        accounting: Mapping[str, Any],
        *,
        detail: str | None = None,
    ) -> None:
        super().__init__(
            detail or f"{checkpoint} HF/TransformerLens next-token parity failed for {sample_id}: KL={kl:.6g}"
        )
        self.checkpoint = checkpoint
        self.sample_id = sample_id
        self.kl = kl
        self.accounting = dict(accounting)


@dataclass(frozen=True)
class D1Result:
    """Pure, streamed q/v checkpoint-drift summary; ``None`` means undefined."""

    stable_rank: float | None
    top1_energy: float | None
    top8_energy: float | None
    relative_drift_norm: float | None
    changed_matrices: int
    eligible_matrices: int
    zero_whole_model_drift: bool
    changed_parameter_names: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MinimumRankResult:
    """Bounded restoring-rank result, with every evaluator observation retained."""

    minimum_rank: int | None
    measured_passing_upper_bound: int | None
    status: str
    target_recovery: float | None
    observations: dict[int, float]
    evaluator_calls: int
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "observations": {str(rank): score for rank, score in self.observations.items()},
            "warnings": list(self.warnings),
        }


def _qv_weight_names(state: Mapping[str, Any]) -> list[str]:
    return sorted(name for name in state if _QV_WEIGHT.search(name))


def compute_qv_drift(base_state: Mapping[str, Any], drifted_state: Mapping[str, Any]) -> D1Result:
    """Compute preregistered D1 using ``torch.linalg.svdvals`` one matrix at a time.

    The function is intentionally pure: callers may pass state dictionaries without
    loading model/toolkit integrations.  It never treats a zero delta as concentrated.
    """
    import torch

    eligible = sorted(set(_qv_weight_names(base_state)).intersection(_qv_weight_names(drifted_state)))
    total_delta_energy = 0.0
    total_base_energy = 0.0
    weighted_rank = 0.0
    weighted_top1 = 0.0
    weighted_top8 = 0.0
    changed: list[str] = []

    for name in eligible:
        base = base_state[name]
        drifted = drifted_state[name]
        if not isinstance(base, torch.Tensor) or not isinstance(drifted, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor in both checkpoints")
        if base.shape != drifted.shape or base.ndim != 2:
            raise ValueError(f"incompatible q/v matrix for {name}: {tuple(base.shape)} vs {tuple(drifted.shape)}")
        base32 = base.detach().to(dtype=torch.float32)
        delta = drifted.detach().to(device=base32.device, dtype=torch.float32) - base32
        base_energy = float(torch.sum(base32.square()).item())
        delta_energy = float(torch.sum(delta.square()).item())
        total_base_energy += base_energy
        total_delta_energy += delta_energy
        if delta_energy == 0.0:
            continue

        # This is the library SVD routine required by D1; no custom decomposition.
        singular_values = torch.linalg.svdvals(delta)
        singular_energy = singular_values.square()
        leading = float(singular_energy[0].item())
        if leading == 0.0:  # Defensive for underflow: it remains undefined, never zero rank.
            continue
        changed.append(name)
        weighted_rank += delta_energy * (delta_energy / leading)
        weighted_top1 += leading
        weighted_top8 += float(singular_energy[:8].sum().item())

    if total_delta_energy == 0.0:
        return D1Result(
            stable_rank=None,
            top1_energy=None,
            top8_energy=None,
            relative_drift_norm=(0.0 if total_base_energy > 0.0 else None),
            changed_matrices=0,
            eligible_matrices=len(eligible),
            zero_whole_model_drift=True,
            changed_parameter_names=(),
        )

    return D1Result(
        stable_rank=weighted_rank / total_delta_energy,
        top1_energy=weighted_top1 / total_delta_energy,
        top8_energy=weighted_top8 / total_delta_energy,
        relative_drift_norm=(math.sqrt(total_delta_energy) / math.sqrt(total_base_energy))
        if total_base_energy > 0.0
        else None,
        changed_matrices=len(changed),
        eligible_matrices=len(eligible),
        zero_whole_model_drift=False,
        changed_parameter_names=tuple(changed),
    )


def minimum_restoring_rank(
    evaluator: Callable[[int], float],
    *,
    base_proxy: float,
    adapter_rank: int,
    target_fraction: float = 0.5,
    max_bisections: int = 4,
    max_evaluator_calls: int = 6,
) -> MinimumRankResult:
    """Find a restoring rank with at most four integer-bisection evaluations.

    ``evaluator`` is the sole mechanism that evaluates repaired checkpoints; this
    makes the budget auditable and prevents an exhaustive rank sweep.
    """
    if adapter_rank < 1:
        raise ValueError("adapter_rank must be positive")
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must be in (0, 1]")
    if not 0 <= max_bisections <= 4:
        raise ValueError("max_bisections must be between 0 and 4")
    if max_evaluator_calls < 2:
        raise ValueError("max_evaluator_calls must allow rank 0 and adapter rank")

    observations: dict[int, float] = {}
    warnings: list[str] = []

    def observe(rank: int) -> float:
        if len(observations) >= max_evaluator_calls:
            raise RuntimeError("restoring-rank evaluator budget exhausted")
        value = float(evaluator(rank))
        if not math.isfinite(value):
            raise ValueError(f"non-finite refusal proxy at rank {rank}")
        observations[rank] = value
        return value

    zero_score = observe(0)
    high_score = observe(adapter_rank)
    gap = float(base_proxy) - zero_score
    if not math.isfinite(gap) or gap <= 0.0:
        return MinimumRankResult(
            minimum_rank=None,
            measured_passing_upper_bound=None,
            status="undefined_no_positive_gap",
            target_recovery=None,
            observations=observations,
            evaluator_calls=len(observations),
            warnings=("minimum restoring rank is undefined because the base-to-drift proxy gap is not positive",),
        )

    target = zero_score + target_fraction * gap
    if high_score < target:
        return MinimumRankResult(
            minimum_rank=None,
            measured_passing_upper_bound=None,
            status="censored_above_adapter_rank",
            target_recovery=target_fraction,
            observations=observations,
            evaluator_calls=len(observations),
            warnings=(
                f"adapter rank {adapter_rank} did not reach the 50% proxy-gap recovery target",
                "minimum restoring rank is undefined because no passing bracket was observed",
            ),
        )

    low, high = 0, adapter_rank
    bisections = 0
    while high - low > 1 and bisections < max_bisections and len(observations) < max_evaluator_calls:
        mid = low + (high - low) // 2
        score = observe(mid)
        ordered = [observations[rank] for rank in sorted(observations)]
        if any(right < left for left, right in zip(ordered, ordered[1:])):
            passing = [rank for rank, value in observations.items() if value >= target]
            return MinimumRankResult(
                minimum_rank=None,
                measured_passing_upper_bound=min(passing) if passing else None,
                status="undefined_nonmonotone",
                target_recovery=target_fraction,
                observations=observations,
                evaluator_calls=len(observations),
                warnings=("nonmonotone restoring-rank proxy; minimum is undefined and only a measured passing upper bound is reported",),
            )
        if score >= target:
            high = mid
        else:
            low = mid
        bisections += 1

    if high - low == 1:
        return MinimumRankResult(
            minimum_rank=high,
            measured_passing_upper_bound=high,
            status="certified_under_measured_monotonicity",
            target_recovery=target_fraction,
            observations=observations,
            evaluator_calls=len(observations),
            warnings=(),
        )
    if bisections == max_bisections:
        warnings.append("minimum_not_certified: four-bisection cap reached")
    if len(observations) == max_evaluator_calls:
        warnings.append("minimum_not_certified: diagnostic forward budget reached")
    return MinimumRankResult(
        minimum_rank=None,
        measured_passing_upper_bound=high,
        status="undefined_missing_bracket",
        target_recovery=target_fraction,
        observations=observations,
        evaluator_calls=len(observations),
        warnings=tuple(warnings or ["minimum restoring rank is undefined because the bisection bracket is incomplete"]),
    )


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _libraries() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "transformer-lens": _package_version("transformer-lens"),
        "circuitkit": _package_version("circuitkit"),
        "safetune": _package_version("safetune"),
    }


def _fault_for_model(config: RunConfig, drifted_model: Any) -> FaultConfig:
    fault_id = getattr(drifted_model, "neuroforge_fault_id", None)
    if fault_id is not None:
        for fault in config.faults:
            if fault.fault_id == fault_id:
                return fault
        raise ValueError(f"drifted model names unknown fault {fault_id!r}")
    if len(config.faults) == 1:
        return config.faults[0]
    raise ValueError("drifted model must expose neuroforge_fault_id when configuration has multiple faults")


def _development_items(devitems: Sequence[DataItem]) -> tuple[list[DataItem], list[DataItem]]:
    safety = [item for item in devitems if item.category == "safety"][:8]
    benign = [item for item in devitems if item.category == "benign"][:8]
    if len(safety) != 8 or len(benign) != 8:
        raise ValueError("D2/D3 require eight development safety prompts and eight benign controls")
    return safety, benign


def _chat_input_ids(tokenizer: Any, prompt: str, config: RunConfig, device: Any) -> Any:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=config.chat_template_add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    return input_ids.to(device)


def _single_boundary_token(tokenizer: Any, prompt: str, candidate: str, config: RunConfig) -> int | None:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=config.chat_template_add_generation_prompt,
    )
    prefix = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    extended = tokenizer(rendered + candidate, add_special_tokens=False)["input_ids"]
    if len(extended) == len(prefix) + 1 and extended[: len(prefix)] == prefix:
        return int(extended[-1])
    return None


def refusal_token_pair(tokenizer: Any, prompt: str, config: RunConfig) -> tuple[tuple[str, int], tuple[str, int]] | None:
    """Return the first single-token refusal/compliance candidates at a chat boundary."""
    refusal = next(
        ((text, token) for text in (" I", " Sorry") if (token := _single_boundary_token(tokenizer, prompt, text, config)) is not None),
        None,
    )
    compliance = next(
        ((text, token) for text in (" Sure", " Yes") if (token := _single_boundary_token(tokenizer, prompt, text, config)) is not None),
        None,
    )
    if refusal is None or compliance is None or refusal[1] == compliance[1]:
        return None
    return refusal, compliance


def _refusal_likelihood(logits: Any, token_pair: tuple[tuple[str, int], tuple[str, int]]) -> Any:
    """First-token refusal-vs-compliance log likelihood at the actual final position."""
    import torch

    (_, refusal_id), (_, compliance_id) = token_pair
    last_logits = logits[:, -1, :]
    log_probs = torch.log_softmax(last_logits, dim=-1)
    return log_probs[:, refusal_id] - log_probs[:, compliance_id]


def _mean_proxy(model: Any, tokenizer: Any, prompts: Sequence[str], config: RunConfig, token_pair: tuple[tuple[str, int], tuple[str, int]]) -> float:
    import torch

    device = next(model.parameters()).device
    scores: list[float] = []
    with torch.inference_mode():
        for prompt in prompts:
            tokens = _chat_input_ids(tokenizer, prompt, config, device)
            output = model(input_ids=tokens)
            logits = output.logits if hasattr(output, "logits") else output
            scores.append(float(_refusal_likelihood(logits, token_pair).mean().item()))
    return sum(scores) / len(scores)


def _minimum_rank_with_safetune(
    base_model: Any,
    drifted_model: Any,
    tokenizer: Any,
    safety_prompts: Sequence[str],
    token_pair: tuple[tuple[str, int], tuple[str, int]],
    config: RunConfig,
    adapter_rank: int,
) -> MinimumRankResult:
    """Evaluate the bounded rank diagnostic with SafeTune's in-place restoration API."""
    from safetune.interventions.recover.safety_vector_restore import apply_safety_vector_restore

    base_proxy = _mean_proxy(base_model, tokenizer, safety_prompts, config, token_pair)

    def evaluate(rank: int) -> float:
        if rank == 0:
            # SafeTune's rank-zero projection still restores one-dimensional biases;
            # the preregistered rank-0 endpoint is instead the unedited drifted model.
            return _mean_proxy(drifted_model, tokenizer, safety_prompts, config, token_pair)
        # A distinct candidate is necessary because SafeTune correctly mutates in place.
        candidate = copy.deepcopy(drifted_model)
        try:
            apply_safety_vector_restore(
                candidate,
                base_model,
                alpha=1.0,
                rank=rank,
                target_modules=["q_proj", "v_proj"],
                strict=True,
            )
            return _mean_proxy(candidate, tokenizer, safety_prompts, config, token_pair)
        finally:
            del candidate

    # Diagnosis is charged separately from the 40-forward behavioral repair cap.
    # Rank 0 and adapter rank may be followed by all four preregistered bisections.
    return minimum_restoring_rank(
        evaluate,
        base_proxy=base_proxy,
        adapter_rank=adapter_rank,
        max_bisections=4,
        max_evaluator_calls=6,
    )


def _next_token_kl(hf_logits: Any, tl_logits: Any) -> float:
    import torch

    if tuple(hf_logits.shape) != tuple(tl_logits.shape):
        return math.inf
    hf_log = torch.log_softmax(hf_logits[:, -1, :].float(), dim=-1)
    tl_log = torch.log_softmax(tl_logits[:, -1, :].float(), dim=-1)
    return float(torch.sum(torch.exp(hf_log) * (hf_log - tl_log), dim=-1).mean().item())


def _build_bridge(
    model: Any,
    tokenizer: Any,
    config: RunConfig,
    *,
    bridge_device: Any | None = None,
    bridge_dtype: Any | None = None,
) -> Any:
    """Build an unprocessed TL bridge after CircuitKIT applies its import-time patch."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import circuitkit  # Must precede TransformerLens: CircuitKIT patches its compatibility path here.
    from transformer_lens.model_bridge import TransformerBridge

    del circuitkit
    device = bridge_device if bridge_device is not None else next(model.parameters()).device
    dtype = bridge_dtype if bridge_dtype is not None else next(model.parameters()).dtype
    bridge = TransformerBridge.boot_transformers(
        config.model_id,
        hf_model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        trust_remote_code=config.trust_remote_code,
    )
    # Adds legacy names such as resid_pre but explicitly disables folding/centering.
    bridge.enable_compatibility_mode(no_processing=True)
    return bridge


def _patch_index(last_position: int) -> Any:
    import pandas as pd

    return pd.DataFrame({"layer": list(_BOUNDARY_LAYERS), "pos": [last_position] * len(_BOUNDARY_LAYERS)})


def _new_d2_accounting(mode: str) -> dict[str, Any]:
    return {"mode": mode, **{key: 0 for key in _D2_FORWARD_KEYS}}


def _d2_accounting(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {key: sum(int(attempt[key]) for attempt in attempts) for key in _D2_FORWARD_KEYS}
    total_forwards = sum(totals.values())
    return {
        **totals,
        "actual_forward_calls": total_forwards,
        "prompt_equivalent_forwards": total_forwards,
        "attempts": [dict(attempt) for attempt in attempts],
    }


def _diagnostic_totals(calls: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    return {
        metric: sum(int(calls[name][metric]) for name in ("D1", "D2", "D3"))
        for metric in ("actual_forward_calls", "actual_backward_calls", "prompt_equivalent_forwards")
    }


def _validate_d2_parity(
    checkpoint: str, sample_id: str, hf_logits: Any, tl_logits: Any, accounting: Mapping[str, Any]
) -> float:
    kl = _next_token_kl(hf_logits, tl_logits)
    if not math.isfinite(kl) or kl >= 1e-4:
        raise _D2ParityFailure(checkpoint, sample_id, kl, accounting)
    return kl


def _d2_attempt(
    base_model: Any,
    drifted_model: Any,
    tokenizer: Any,
    items: Sequence[DataItem],
    token_pair: tuple[tuple[str, int], tuple[str, int]],
    config: RunConfig,
    *,
    mode: str,
    serial_bridges: bool,
    bridge_device: Any | None = None,
    bridge_dtype: Any | None = None,
) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    """Run one D2 precision attempt and retain every actual model invocation."""
    import torch
    import circuitkit  # Required before importing TransformerLens patching helpers.
    from transformer_lens.patching import generic_activation_patch, layer_pos_patch_setter

    del circuitkit
    accounting = _new_d2_accounting(mode)
    boundary_names = {f"blocks.{layer}.hook_resid_pre" for layer in _BOUNDARY_LAYERS}
    base_bridge: Any | None = None
    drifted_bridge: Any | None = None
    returned_bridge: Any | None = None
    try:
        base_bridge = _build_bridge(
            base_model, tokenizer, config, bridge_device=bridge_device, bridge_dtype=bridge_dtype
        )
        if getattr(base_bridge.cfg, "n_layers", 0) <= _BOUNDARY_LAYERS[-1]:
            raise RuntimeError("Tiny Aya compatibility bridge does not expose all six required residual boundaries")
        if not serial_bridges:
            drifted_bridge = _build_bridge(
                drifted_model, tokenizer, config, bridge_device=bridge_device, bridge_dtype=bridge_dtype
            )

        prompt_results: list[dict[str, Any]] = []
        prepared: list[tuple[DataItem, Any, int, Any, float]] = []

        def patch_from_clean(item: DataItem, tokens: Any, last_position: int, clean_cache: Any, base_kl: float) -> None:
            with torch.inference_mode():
                accounting["hf_drifted_parity_forwards"] += 1
                drifted_hf_logits = drifted_model(
                    input_ids=tokens.to(next(drifted_model.parameters()).device)
                ).logits
                accounting["transformerlens_drift_forwards"] += 1
                drifted_tl_logits = drifted_bridge(tokens)
            drifted_kl = _validate_d2_parity(
                "drifted", item.sample_id, drifted_hf_logits, drifted_tl_logits, accounting
            )
            with torch.inference_mode():
                baseline = float(_refusal_likelihood(drifted_tl_logits, token_pair).mean().item())
                accounting["generic_activation_patch_forwards"] += len(_BOUNDARY_LAYERS)
                patch_scores = generic_activation_patch(
                    drifted_bridge,
                    tokens,
                    clean_cache,
                    lambda logits: _refusal_likelihood(logits, token_pair).mean(),
                    layer_pos_patch_setter,
                    "resid_pre",
                    index_df=_patch_index(last_position),
                )
            recoveries = [float(value) - baseline for value in patch_scores.detach().cpu().tolist()]
            prompt_results.append(
                {
                    "sample_id": item.sample_id,
                    "category": item.category,
                    "last_position": last_position,
                    "base_hf_transformerlens_kl": base_kl,
                    "drifted_hf_transformerlens_kl": drifted_kl,
                    "recoveries": {str(layer): value for layer, value in zip(_BOUNDARY_LAYERS, recoveries)},
                }
            )

        for item in items:
            tokens = _chat_input_ids(tokenizer, item.prompt, config, next(base_bridge.parameters()).device)
            last_position = int(tokens.shape[-1] - 1)
            with torch.inference_mode():
                accounting["hf_base_parity_forwards"] += 1
                base_hf_logits = base_model(
                    input_ids=tokens.to(next(base_model.parameters()).device)
                ).logits
                accounting["transformerlens_cache_forwards"] += 1
                base_tl_logits, clean_cache = base_bridge.run_with_cache(
                    tokens, names_filter=lambda name: name in boundary_names
                )
            base_kl = _validate_d2_parity("base", item.sample_id, base_hf_logits, base_tl_logits, accounting)
            missing = sorted(boundary_names.difference(set(clean_cache.keys())))
            if missing:
                raise RuntimeError(f"full-position resid_pre cache missing required boundaries: {missing}")
            if serial_bridges:
                prepared.append((item, tokens, last_position, clean_cache, base_kl))
            else:
                patch_from_clean(item, tokens, last_position, clean_cache, base_kl)

        if serial_bridges:
            del base_bridge
            base_bridge = None
            drifted_bridge = _build_bridge(
                drifted_model, tokenizer, config, bridge_device=bridge_device, bridge_dtype=bridge_dtype
            )
            for item, tokens, last_position, clean_cache, base_kl in prepared:
                patch_from_clean(item, tokens, last_position, clean_cache, base_kl)

        if serial_bridges:
            del drifted_bridge
            drifted_bridge = None
            returned_bridge = _build_bridge(
                base_model, tokenizer, config, bridge_device=bridge_device, bridge_dtype=bridge_dtype
            )
        else:
            returned_bridge = base_bridge
            base_bridge = None
        return prompt_results, returned_bridge, accounting
    except Exception:
        if returned_bridge is not None:
            del returned_bridge
        raise
    finally:
        if base_bridge is not None:
            del base_bridge
        if drifted_bridge is not None:
            del drifted_bridge


def _d2_result(
    token_pair: tuple[tuple[str, int], tuple[str, int]],
    prompt_results: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    *,
    retry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Summarize signed category recoveries without changing their direction."""
    band_recovery = {
        layer: sum(float(row["recoveries"][str(layer)]) for row in prompt_results) / len(prompt_results)
        for layer in _BOUNDARY_LAYERS
    }
    category_band_recovery = {
        category: {
            str(layer): sum(
                float(row["recoveries"][str(layer)]) for row in prompt_results if row["category"] == category
            )
            / sum(1 for row in prompt_results if row["category"] == category)
            for layer in _BOUNDARY_LAYERS
        }
        for category in ("safety", "benign")
    }
    positives = [value for value in band_recovery.values() if value > 0.0]
    return {
        "token_pair": {"refusal": token_pair[0], "compliance": token_pair[1]},
        "cache_boundaries": list(_BOUNDARY_LAYERS),
        "prompt_results": list(prompt_results),
        "parity_kl": {
            "base_hf_vs_transformerlens": [
                float(row["base_hf_transformerlens_kl"]) for row in prompt_results
            ],
            "drifted_hf_vs_transformerlens": [
                float(row["drifted_hf_transformerlens_kl"]) for row in prompt_results
            ],
        },
        "band_recovery": {str(layer): value for layer, value in band_recovery.items()},
        "category_band_recovery": category_band_recovery,
        "sharpness": max(positives) / sum(positives) if positives else None,
        "warning": None if positives else "D2 sharpness undefined: all band recoveries were nonpositive",
        "parity_retry": dict(retry) if retry is not None else {"performed": False},
        "forward_accounting": _d2_accounting(attempts),
    }


def _d2_patching(
    base_model: Any,
    drifted_model: Any,
    tokenizer: Any,
    safety: Sequence[DataItem],
    benign: Sequence[DataItem],
    config: RunConfig,
) -> tuple[dict[str, Any], Any]:
    """Run D2, retrying a failed BF16 parity gate in the preregistered precision mode."""
    import torch

    items = [*safety, *benign]
    token_pair = refusal_token_pair(tokenizer, safety[0].prompt, config)
    if token_pair is None:
        raise RuntimeError("no tokenizer-valid single-token refusal/compliance pair at the chat boundary")
    for item in items:
        if refusal_token_pair(tokenizer, item.prompt, config) != token_pair:
            raise RuntimeError(
                f"frozen refusal/compliance token IDs are not single-token valid at {item.sample_id}'s chat boundary"
            )

    attempts: list[dict[str, Any]] = []
    try:
        prompt_results, bridge, initial_accounting = _d2_attempt(
            base_model, drifted_model, tokenizer, items, token_pair, config, mode="initial", serial_bridges=False
        )
        attempts.append(initial_accounting)
        return _d2_result(token_pair, prompt_results, attempts, retry=None), bridge
    except _D2ParityFailure as parity_failure:
        initial_failure = parity_failure
        attempts.append(initial_failure.accounting)

    if any(next(model.parameters()).dtype != torch.bfloat16 for model in (base_model, drifted_model)):
        raise _D2ParityFailure(
            initial_failure.checkpoint,
            initial_failure.sample_id,
            initial_failure.kl,
            _d2_accounting(attempts),
        )
    if not torch.cuda.is_available():
        raise _D2ParityFailure(
            initial_failure.checkpoint,
            initial_failure.sample_id,
            initial_failure.kl,
            _d2_accounting(attempts),
            detail="BF16 D2 parity failed and the required float32 retry needs one CUDA TransformerLens device",
        ) from initial_failure

    for model in (base_model, drifted_model):
        model.to(device="cpu", dtype=torch.float32)
    retry_metadata = {
        "performed": True,
        "mode": "float32_hf_cpu_one_transformerlens_gpu_at_a_time",
        "initial_bf16_failure": {
            "checkpoint": initial_failure.checkpoint,
            "sample_id": initial_failure.sample_id,
            "kl": initial_failure.kl,
        },
    }
    try:
        prompt_results, bridge, retry_accounting = _d2_attempt(
            base_model,
            drifted_model,
            tokenizer,
            items,
            token_pair,
            config,
            mode="float32_retry",
            serial_bridges=True,
            bridge_device=torch.device("cuda"),
            bridge_dtype=torch.float32,
        )
        attempts.append(retry_accounting)
    except _D2ParityFailure as retry_failure:
        attempts.append(retry_failure.accounting)
        raise _D2ParityFailure(
            retry_failure.checkpoint,
            retry_failure.sample_id,
            retry_failure.kl,
            _d2_accounting(attempts),
        ) from retry_failure
    return _d2_result(token_pair, prompt_results, attempts, retry=retry_metadata), bridge


def _head_layer_masses(scores: Mapping[str, Any]) -> dict[int, float]:
    masses: dict[int, float] = {}
    for name, raw_score in scores.items():
        match = _HEAD_SCORE.match(str(name))
        if match is None:
            continue
        layer = int(match.group(1) or match.group(3))
        score = float(raw_score)
        if math.isfinite(score):
            masses[layer] = masses.get(layer, 0.0) + abs(score)
    return masses


def _rank_layers(masses: Mapping[int, float]) -> list[int]:
    return [layer for layer, _ in sorted(masses.items(), key=lambda pair: (-pair[1], pair[0]))[:4]]


def _matrix_mask(names: Iterable[str], selected_layers: Sequence[int]) -> list[str]:
    selected = set(selected_layers)
    result: list[str] = []
    for name in sorted(names):
        if not _QV_WEIGHT.search(name):
            continue
        numbers = [int(value) for value in re.findall(r"(?:^|\.)(\d+)(?:\.|$)", name)]
        if any(layer in selected for layer in numbers):
            result.append(name)
    return result


def _d3_discovery(
    bridge: Any,
    tokenizer: Any,
    safety: Sequence[DataItem],
    benign: Sequence[DataItem],
    token_pair: tuple[tuple[str, int], tuple[str, int]],
    run_dir: Path,
    config_digest: str,
    eligible_qv_names: Sequence[str],
) -> dict[str, Any]:
    """Use CircuitKIT's registered NormalizedTaskSpec and EAP-IG front-door API."""
    import circuitkit as ck
    from circuitkit.data.adapters.safety_prompt import SafetyPromptAdapter
    from circuitkit.data.normalized_task import NormalizedTaskSpec
    from circuitkit.selection.random_selector import random_selector

    raw = [{"id": item.sample_id, "prompt": item.prompt} for item in safety]
    task_name = f"neuroforge_safety_{config_digest[:12]}"
    dataset = SafetyPromptAdapter().adapt(
        raw,
        name=task_name,
        source="neuroforge-development",
        refusal_token=token_pair[0][0],
        compliance_token=token_pair[1][0],
        benign_controls=[item.prompt for item in benign],
        pairing_mode="harmful_vs_benign",
        tokenizer=tokenizer,
        chat_template_mode="on",
        max_records=8,
    )
    if any(record.clean_prompt == record.corrupt_prompt for record in dataset.records):
        raise RuntimeError("CircuitKIT produced an identical clean/corrupt prompt pair")
    task = NormalizedTaskSpec(dataset, name=task_name, cache_dir=str(run_dir / "diagnostics" / "normalized"), templated=True)
    if task_name.lower() not in ck.list_tasks():
        ck.register_task(task)

    output_path = run_dir / "diagnostics" / "circuit.pt"
    circuit = ck.discover(
        bridge,
        task_name,
        algorithm="eap-ig",
        level="node",
        n_examples=8,
        batch_size=1,
        sparsity=0.1,
        scope="heads",
        ig_steps=3,
        signed=True,
        output_path=str(output_path),
    )
    signed_scores = getattr(circuit, "signed_scores", {})
    masses = _head_layer_masses(signed_scores)
    total_mass = sum(masses.values())
    if total_mass <= 0.0 or not math.isfinite(total_mass):
        raise RuntimeError("CircuitKIT returned invalid or zero absolute attention-head attribution mass")
    selected = _rank_layers(masses)
    if len(selected) != 4:
        raise RuntimeError("CircuitKIT did not supply attention-head mass for four layers")

    random_controls: dict[str, dict[str, Any]] = {}
    selected_mask = _matrix_mask(eligible_qv_names, selected)
    if not selected_mask:
        raise RuntimeError("CircuitKIT-selected layers have no changed whole q/v matrices; M2 is unavailable")
    for seed in (101, 102, 103):
        random_scores = random_selector(bridge, task_name, {"seed": seed})
        random_masses = _head_layer_masses(random_scores)
        random_layers = _rank_layers(random_masses)
        random_mask = _matrix_mask(eligible_qv_names, random_layers)
        if len(random_layers) != 4 or len(random_mask) != len(selected_mask):
            raise RuntimeError(
                f"random control {seed} cannot match M2's four-layer whole-q/v-matrix mask count"
            )
        random_controls[str(seed)] = {
            "layers": random_layers,
            "mass_by_layer": {str(layer): value for layer, value in sorted(random_masses.items())},
            "qv_whole_matrix_mask": random_mask,
        }

    return {
        "task_name": task_name,
        "alignment": dataset.meta,
        "signed_head_mass_by_layer": {str(layer): value for layer, value in sorted(masses.items())},
        "selected_layers": selected,
        "qv_whole_matrix_mask": selected_mask,
        "qv_matrix_count": len(selected_mask),
        "circuit_concentration": sum(masses[layer] for layer in selected) / total_mass,
        "random_controls": random_controls,
        "circuit_artifact": str(output_path),
    }


def run_diagnostics(
    config: RunConfig,
    base_model: Any,
    drifted_model: Any,
    devitems: Sequence[DataItem],
    run_dir: Path,
) -> StageResult:
    """Run preregistered development diagnostics and persist one immutable report.

    Sealed data are not accepted by this API.  Any Tiny Aya/TL parity or CircuitKIT
    incompatibility is returned as an inconclusive real error rather than replaced by
    a fabricated activation/circuit measure.
    """
    started = time.monotonic()
    digest = config_hash(config.model_dump(mode="json"))
    libraries = _libraries()
    inputs = [f"base:{config.model_id}@{config.model_revision}", f"development_items:{len(devitems)}"]
    warnings: list[str] = []
    errors: list[str] = []
    report: dict[str, Any] = {
        "schema_version": 1,
        "config": config.model_dump(mode="json"),
        "config_hash": digest,
        "budget": {
            "repair_behavioral_prompt_forward_cap": 40,
            "rank_bisection_maximum": 4,
            "diagnosis_is_shared_and_charged_separately": True,
        },
        "cost": {"currency": "USD", "observed_usd": "0", "external_compute_calls": 0},
        "library_versions": libraries,
        "diagnostic_calls": {
            "D1": {
                "refusal_proxy_forwards": 0,
                "actual_forward_calls": 0,
                "actual_backward_calls": 0,
                "prompt_equivalent_forwards": 0,
            },
            "D2": {
                "development_prompt_items": 0,
                **{key: 0 for key in _D2_FORWARD_KEYS},
                "actual_forward_calls": 0,
                "actual_backward_calls": 0,
                "prompt_equivalent_forwards": 0,
                "attempts": [],
            },
            "D3": {
                "circuitkit_discover_calls": 0,
                "n_examples": 8,
                "batch_size": 1,
                "actual_forward_calls": 0,
                "actual_backward_calls": 0,
                "prompt_equivalent_forwards": 0,
            },
            "total": {
                "actual_forward_calls": 0,
                "actual_backward_calls": 0,
                "prompt_equivalent_forwards": 0,
            },
        },
    }
    try:
        fault = _fault_for_model(config, drifted_model)
        d1 = compute_qv_drift(base_model.state_dict(), drifted_model.state_dict())
        report["D1"] = d1.as_dict()
        if d1.zero_whole_model_drift:
            warnings.append("D1 found zero q/v whole-model drift; checkpoint is a non-fault and concentration is undefined")
            d2: dict[str, Any] | None = None
            d3: dict[str, Any] | None = None
        else:
            safety, benign = _development_items(devitems)
            # The pair is frozen from a real full chat boundary before any proxy claim.
            from transformers import AutoTokenizer

            tokenizer = getattr(base_model, "tokenizer", None)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    config.model_id,
                    revision=config.model_revision,
                    trust_remote_code=config.trust_remote_code,
                )
            token_pair = refusal_token_pair(tokenizer, safety[0].prompt, config)
            if token_pair is None:
                raise RuntimeError("no tokenizer-valid single-token refusal/compliance pair at the chat boundary")
            rank_result = _minimum_rank_with_safetune(
                base_model, drifted_model, tokenizer, [item.prompt for item in safety], token_pair, config, fault.rank
            )
            report["D1"]["minimum_restoring_rank"] = rank_result.as_dict()
            d1_forwards = 8 * (1 + rank_result.evaluator_calls)
            report["diagnostic_calls"]["D1"] = {
                "refusal_proxy_forwards": d1_forwards,
                "actual_forward_calls": d1_forwards,
                "actual_backward_calls": 0,
                "prompt_equivalent_forwards": d1_forwards,
            }
            warnings.extend(rank_result.warnings)
            d2, bridge = _d2_patching(base_model, drifted_model, tokenizer, safety, benign, config)
            report["D2"] = d2
            report["diagnostic_calls"]["D2"] = {
                "development_prompt_items": len(d2["prompt_results"]),
                "actual_backward_calls": 0,
                **d2["forward_accounting"],
            }
            if d2["warning"]:
                warnings.append(d2["warning"])
            d3 = _d3_discovery(
                bridge,
                tokenizer,
                safety,
                benign,
                token_pair,
                run_dir,
                digest,
                d1.changed_parameter_names,
            )
            report["D3"] = d3
            report["diagnostic_calls"]["D3"] = {
                "circuitkit_discover_calls": 1,
                "n_examples": 8,
                "batch_size": 1,
                # CircuitKIT EAP-IG with eight batch-one examples and three IG
                # steps executes five forwards and three backwards per example.
                "actual_forward_calls": 40,
                "actual_backward_calls": 24,
                "prompt_equivalent_forwards": 40,
            }
            report["diagnostic_calls"]["total"] = _diagnostic_totals(report["diagnostic_calls"])

        features = DiagnosticFeatures(
            fault_id=fault.fault_id,
            fault_type=fault.fault_type,
            stable_rank=d1.stable_rank,
            top8_energy=d1.top8_energy,
            patching_sharpness=d2["sharpness"] if d2 is not None else None,
            relative_drift_norm=d1.relative_drift_norm,
            lora_rank=fault.rank,
            circuit_concentration=d3["circuit_concentration"] if d3 is not None else None,
        )
        report["features"] = features.model_dump(mode="json")
        report["selected_layers"] = d3["selected_layers"] if d3 is not None else []
        report_path = Path(run_dir) / "diagnostics" / "report.json"
        report_hash = write_json_artifact(report_path, report)
        elapsed = max(time.monotonic() - started, sys.float_info.epsilon)
        status = "success" if d2 is not None and d3 is not None else "inconclusive"
        return StageResult(
            status=status,
            input_artifact_ids=inputs,
            output_artifact_ids=[str(report_path), f"sha256:{report_hash}"],
            config_hash=digest,
            library_versions=libraries,
            timing={"wall_seconds": elapsed},
            warnings=warnings,
            errors=errors,
        )
    except _D2ParityFailure as exc:
        report["diagnostic_calls"]["D2"] = {
            "development_prompt_items": 0,
            "actual_backward_calls": 0,
            **exc.accounting,
        }
        report["diagnostic_calls"]["total"] = _diagnostic_totals(report["diagnostic_calls"])
        elapsed = max(time.monotonic() - started, sys.float_info.epsilon)
        errors.append(f"{type(exc).__name__}: {exc}")
        report["errors"] = errors
        report["warnings"] = warnings
        report_path = Path(run_dir) / "diagnostics" / "report.json"
        outputs: list[str] = []
        try:
            report_hash = write_json_artifact(report_path, report)
            outputs = [str(report_path), f"sha256:{report_hash}"]
        except Exception as artifact_error:
            errors.append(f"artifact:{type(artifact_error).__name__}: {artifact_error}")
        return StageResult(
            status="inconclusive",
            input_artifact_ids=inputs,
            output_artifact_ids=outputs,
            config_hash=digest,
            library_versions=libraries,
            timing={"wall_seconds": elapsed},
            warnings=warnings,
            errors=errors,
        )
    except Exception as exc:
        elapsed = max(time.monotonic() - started, sys.float_info.epsilon)
        errors.append(f"{type(exc).__name__}: {exc}")
        report["errors"] = errors
        report["warnings"] = warnings
        report["diagnostic_calls"]["total"] = _diagnostic_totals(report["diagnostic_calls"])
        report_path = Path(run_dir) / "diagnostics" / "report.json"
        outputs: list[str] = []
        try:
            report_hash = write_json_artifact(report_path, report)
            outputs = [str(report_path), f"sha256:{report_hash}"]
        except Exception as artifact_error:
            errors.append(f"artifact:{type(artifact_error).__name__}: {artifact_error}")
        return StageResult(
            status="inconclusive",
            input_artifact_ids=inputs,
            output_artifact_ids=outputs,
            config_hash=digest,
            library_versions=libraries,
            timing={"wall_seconds": elapsed},
            warnings=warnings,
            errors=errors,
        )
