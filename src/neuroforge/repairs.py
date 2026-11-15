# WHY CUSTOM: SafeTune supplies the repair operators, while this module enforces the
# preregistered one-attempt masks, immutable references, and reloadable artifacts.
"""One-shot SafeTune repair candidates and integrity-checked reconstruction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .artifacts import (
    ArtifactIntegrityError,
    read_manifest,
    read_tensor_artifact,
    sha256_file,
    verify_manifest,
    write_json_artifact,
    write_manifest,
    write_tensor_artifact,
)
from .contracts import (
    DataItem,
    FaultConfig,
    GateDecision,
    GateOutcome,
    Metrics,
    PatchManifest,
    RunConfig,
    StageResult,
    config_hash,
)

RepairMethod = Literal["B0", "M1", "M2", "M3", "R0", "R1", "R2"]
_RANDOM_SEEDS = {"R0": 101, "R1": 102, "R2": 103}
_M3_FALLBACK_LAYERS = [16, 17, 18, 19, 20]


class RepairUnavailable(RuntimeError):
    """Raised when frozen repair inputs or an upstream API are unavailable."""


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _unquantized_base(config: RunConfig) -> Any:
    """Load a fresh editable checkpoint; packed bitsandbytes tensors are never repaired."""
    import torch
    from transformers import AutoModelForCausalLM

    from .modeling import _source

    source, revision, local_files_only, _ = _source(config, None)
    dtype = torch.float16 if config.cloud_dtype == "fp16" else torch.bfloat16
    return AutoModelForCausalLM.from_pretrained(
        source,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=config.trust_remote_code,
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )


def _merged_drift(config: RunConfig, adapter_path: Path) -> Any:
    from peft import PeftModel

    if not adapter_path.is_dir():
        raise RepairUnavailable(f"missing PEFT adapter directory: {adapter_path}")
    base = _unquantized_base(config)
    adapted = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    merger = getattr(adapted, "merge_and_unload", None)
    if not callable(merger):
        raise RepairUnavailable("installed PEFT adapter does not expose merge_and_unload")
    merged = merger(safe_merge=True)
    merged.eval()
    return merged


def _layer_index(name: str) -> int | None:
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return None


def _eligible_names(model: Any, layers: Sequence[int]) -> list[str]:
    selected = set(int(layer) for layer in layers)
    names = []
    for name, tensor in model.state_dict().items():
        if not name.endswith(".weight") or not any(module in name for module in ("q_proj", "v_proj")):
            continue
        if _layer_index(name) in selected:
            if tensor.ndim != 2:
                raise RepairUnavailable(f"selected q/v tensor is not a matrix: {name}")
            names.append(name)
    if len(names) != 2 * len(selected):
        raise RepairUnavailable(
            f"circuit mask must select exactly q/v matrices for every layer; got {len(names)} for {sorted(selected)}"
        )
    return sorted(names)


def _d3_diagnostics(diagnostics: Mapping[str, Any]) -> Mapping[str, Any]:
    d3 = diagnostics.get("D3")
    if d3 is None:
        return diagnostics
    if not isinstance(d3, Mapping):
        raise RepairUnavailable("diagnostics.D3 must be an object")
    return d3


def _four_distinct_layers(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 4 or not all(isinstance(layer, int) for layer in value):
        raise RepairUnavailable(f"{field} must contain exactly four discovered integer layers")
    if len(set(value)) != 4:
        raise RepairUnavailable(f"{field} must not repeat a layer")
    return value


def _circuit_layers(diagnostics: Mapping[str, Any]) -> list[int]:
    d3 = _d3_diagnostics(diagnostics)
    report_layers = diagnostics.get("selected_layers")
    d3_layers = d3.get("selected_layers")
    if report_layers is not None and d3_layers is not None and report_layers != d3_layers:
        raise RepairUnavailable("diagnostics.selected_layers and diagnostics.D3.selected_layers disagree")
    value = d3_layers if d3_layers is not None else report_layers
    return _four_distinct_layers(value, "diagnostics.D3.selected_layers")


def _random_layers(diagnostics: Mapping[str, Any], method: str) -> list[int]:
    d3 = _d3_diagnostics(diagnostics)
    controls = d3.get("random_controls")
    seed = _RANDOM_SEEDS[method]
    if not isinstance(controls, Mapping):
        raise RepairUnavailable("random controls require diagnostics.D3.random_controls")
    control = controls.get(str(seed), controls.get(seed))
    if not isinstance(control, Mapping):
        raise RepairUnavailable(f"{method} requires recorded CircuitKIT random control for seed {seed}")
    return _four_distinct_layers(control.get("layers"), f"diagnostics.D3.random_controls[{seed}].layers")


def _expected_qv_mask(diagnostics: Mapping[str, Any], method: str, selected_names: Sequence[str]) -> None:
    d3 = _d3_diagnostics(diagnostics)
    if method == "M2":
        value = d3.get("qv_whole_matrix_mask")
        field = "diagnostics.D3.qv_whole_matrix_mask"
    else:
        seed = _RANDOM_SEEDS[method]
        controls = d3.get("random_controls")
        control = controls.get(str(seed), controls.get(seed)) if isinstance(controls, Mapping) else None
        value = control.get("qv_whole_matrix_mask") if isinstance(control, Mapping) else None
        field = f"diagnostics.D3.random_controls[{seed}].qv_whole_matrix_mask"
    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        raise RepairUnavailable(f"{field} must be a list of q/v tensor names")
    if len(value) != len(set(value)) or sorted(value) != sorted(selected_names):
        raise RepairUnavailable(f"{field} must exactly match the selected q/v tensor capacity")


def _changed_tensors(
    before: Mapping[str, Any], after: Mapping[str, Any], names: Sequence[str]
) -> tuple[dict[str, np.ndarray], dict[str, str], int]:
    """Bridge tensors through float32 safetensors while retaining source dtypes in the manifest."""
    import torch

    changed: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    changed_scalar_count = 0
    for name in names:
        if name not in before or name not in after:
            raise RepairUnavailable(f"repair lost selected parameter {name}")
        if before[name].shape != after[name].shape or before[name].dtype != after[name].dtype:
            raise RepairUnavailable(f"repair changed tensor schema for {name}")
        changed_entries = int(torch.count_nonzero(before[name] != after[name]).item())
        if changed_entries:
            changed[name] = after[name].detach().cpu().float().numpy().copy()
            dtypes[name] = str(after[name].dtype)
            changed_scalar_count += changed_entries
    if not changed:
        raise RepairUnavailable("SafeTune candidate changed no selected tensors")
    return changed, dtypes, changed_scalar_count


def _selected_scalar_capacity(state: Mapping[str, Any], names: Sequence[str]) -> int:
    return sum(int(state[name].numel()) for name in names)


def _rounded_delta_ranks(
    before: Mapping[str, Any], after: Mapping[str, Any], names: Sequence[str]
) -> dict[str, int]:
    import torch

    return {
        name: int(torch.linalg.matrix_rank(after[name].detach().float().cpu() - before[name].detach().float().cpu()).item())
        for name in names
    }


def _model_state_sha256(model: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


def _apply_checked_safety_vector_restore(model: Any, aligned: Any) -> None:
    import safetune.interventions.recover.safety_vector_restore as restore

    messages: list[str] = []

    class _FallbackCapture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if "SVD failed" in message and "using full delta" in message:
                messages.append(message)

    logger = logging.getLogger(restore.__name__)
    prior_level = logger.level
    capture = _FallbackCapture(level=logging.DEBUG)
    logger.addHandler(capture)
    logger.setLevel(logging.DEBUG)
    try:
        restore.apply_safety_vector_restore(
            model,
            aligned,
            alpha=1.0,
            rank=8,
            target_modules=["q_proj", "v_proj"],
            strict=True,
        )
    finally:
        logger.removeHandler(capture)
        logger.setLevel(prior_level)
    if messages:
        raise RepairUnavailable(f"SafeTune truncated SVD fell back to a full delta: {messages[0]}")


def _candidate_manifest(
    config: RunConfig,
    fault: FaultConfig,
    method: str,
    artifact: Path,
    artifact_hash: str,
    changed: int,
    parameters: dict[str, Any],
) -> PatchManifest:
    return PatchManifest(
        base_model=config.model_id,
        base_revision=config.model_revision,
        fault_id=fault.fault_id,
        method=method,
        parameters=parameters,
        library_versions={name: _version(name) for name in ("torch", "safetune", "peft", "circuitkit")},
        artifact_path=artifact.name,
        artifact_sha256=artifact_hash,
        changed_parameter_count=changed,
        validation_metrics=Metrics(),
        decision=GateDecision(decision=GateOutcome.INCONCLUSIVE, reasons=["candidate not independently evaluated"]),
        rollback_reference="reload immutable fault checkpoint and reapply verified artifact",
    )


def _write_budget(destination: Path, method: str, development_forwards_used: int = 0) -> str:
    return write_json_artifact(
        destination / "condition-cost.json",
        {
            "method": method,
            "candidate_attempts": 1,
            "development_forward_budget": 40,
            "development_forwards_used": development_forwards_used,
            "cost_usd": None,
            "cost_status": "not measured",
        },
    )


def _repair_error(config: RunConfig, started: float, destination: Path, error: Exception) -> StageResult:
    error_hash = write_json_artifact(destination / "error.json", {"error": f"{type(error).__name__}: {error}"})
    return StageResult(
        status="blocked" if isinstance(error, RepairUnavailable) else "error",
        input_artifact_ids=[],
        output_artifact_ids=[error_hash],
        config_hash=config_hash(config.model_dump(mode="json")),
        library_versions={name: _version(name) for name in ("torch", "safetune", "peft", "circuitkit")},
        timing={"wall_seconds": max(time.monotonic() - started, 1e-9)},
        errors=[f"{type(error).__name__}: {error}"],
    )


def _chat_token_count(tokenizer: Any, prompt: str, config: RunConfig) -> int:
    # SafeTune's extractor applies this exact raw-prompt chat template once.
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise RepairUnavailable("chat template did not produce input_ids for M3")
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    if isinstance(input_ids, list) and input_ids:
        return len(input_ids[0] if isinstance(input_ids[0], list) else input_ids)
    raise RepairUnavailable("chat template produced an unsupported input_ids shape for M3")


def _m3_prompts(tokenizer: Any, development_items: Sequence[DataItem], config: RunConfig) -> tuple[list[str], list[str]]:
    if config.input_max_tokens != 512:
        raise RepairUnavailable("M3 requires the preregistered 512-token full-chat input cap")
    harmful = [item.prompt for item in development_items if item.category == "safety"]
    harmless = [item.prompt for item in development_items if item.category == "benign"]
    if len(harmful) != 20 or len(harmless) != 20:
        raise RepairUnavailable("M3 requires exactly 20 development safety prompts and 20 development benign prompts")
    for prompt in [*harmful, *harmless]:
        if _chat_token_count(tokenizer, prompt, config) > 512:
            raise RepairUnavailable("M3 development prompt exceeds the frozen full-chat input cap")
    # The library extractor receives raw prompts and applies this same chat template once.
    return harmful, harmless


def repair_candidate(
    config: RunConfig,
    fault: FaultConfig,
    method: RepairMethod,
    adapter_path: Path,
    run_dir: Path,
    *,
    diagnostics: Mapping[str, Any],
    development_items: Sequence[DataItem] = (),
) -> StageResult:
    """Materialize one predeclared repair condition from a fresh merged drift checkpoint."""
    if method not in {"B0", "M1", "M2", "M3", "R0", "R1", "R2"}:
        raise ValueError(f"unknown repair method: {method}")
    started = time.monotonic()
    destination = run_dir / "repairs" / fault.fault_id / method
    destination.mkdir(parents=True, exist_ok=True)
    try:
        if not adapter_path.is_dir():
            raise RepairUnavailable(f"missing PEFT adapter directory: {adapter_path}")
        fault_adapter_hashes = {
            str(path.relative_to(adapter_path)): sha256_file(path)
            for path in adapter_path.rglob("*")
            if path.is_file()
        }
        budget_hash = _write_budget(destination, method, 4 if method == "M3" else 0)
        if method == "B0":
            # Baseline deliberately has no patch and is reconstructed from the immutable fault adapter.
            artifact = destination / "baseline.json"
            artifact_hash = write_json_artifact(
                artifact,
                {"method": "B0", "adapter_path": str(adapter_path), "adapter_hashes": fault_adapter_hashes},
            )
            manifest = _candidate_manifest(
                config,
                fault,
                method,
                artifact,
                artifact_hash,
                0,
                {"untouched": True, "changed_tensor_count": 0, "selected_scalar_capacity": 0, "selected_tensor_names": []},
            )
        elif method == "M1":
            drift = _merged_drift(config, adapter_path)
            aligned = _unquantized_base(config)
            names = sorted(
                name
                for name in drift.state_dict()
                if name.endswith(".weight") and any(module in name for module in ("q_proj", "v_proj"))
            )
            before = {name: drift.state_dict()[name].detach().clone() for name in names}
            _apply_checked_safety_vector_restore(drift, aligned)
            after = drift.state_dict()
            changed, tensor_dtypes, changed_scalar_count = _changed_tensors(before, after, names)
            artifact = destination / "patch.safetensors"
            artifact_hash = write_tensor_artifact(artifact, changed)
            manifest = _candidate_manifest(
                config,
                fault,
                method,
                artifact,
                artifact_hash,
                changed_scalar_count,
                {
                    "alpha": 1.0,
                    "requested_projection_rank": 8,
                    "measured_rounded_delta_rank": _rounded_delta_ranks(before, after, changed),
                    "target_modules": ["q_proj", "v_proj"],
                    "strict": True,
                    "tensor_dtypes": tensor_dtypes,
                    "changed_tensor_count": len(changed),
                    "selected_tensor_names": names,
                    "selected_scalar_capacity": _selected_scalar_capacity(before, names),
                },
            )
        elif method in {"M2", "R0", "R1", "R2"}:
            from safetune.interventions.recover.ctheta import apply_ctheta_from_state_dicts

            layers = _circuit_layers(diagnostics) if method == "M2" else _random_layers(diagnostics, method)
            target = _merged_drift(config, adapter_path)
            immutable_drift = _merged_drift(config, adapter_path)
            immutable_base = _unquantized_base(config)
            names = _eligible_names(target, layers)
            _expected_qv_mask(diagnostics, method, names)
            before = {name: target.state_dict()[name].detach().clone() for name in names}
            positive_sd = {name: immutable_base.state_dict()[name].detach().cpu() for name in names}
            negative_sd = {name: immutable_drift.state_dict()[name].detach().cpu() for name in names}
            apply_ctheta_from_state_dicts(
                target,
                positive_sd,
                negative_sd,
                circuit_info=None,
                strength=1.0,
                layer_subset=layers,
                target_modules=["q_proj", "v_proj"],
            )
            changed, tensor_dtypes, changed_scalar_count = _changed_tensors(before, target.state_dict(), names)
            if set(changed) != set(names):
                raise RepairUnavailable("C-ΔΘ did not change every masked q/v matrix")
            artifact = destination / "patch.safetensors"
            artifact_hash = write_tensor_artifact(artifact, changed)
            parameters = {
                "strength": 1.0,
                "layers": layers,
                "target_modules": ["q_proj", "v_proj"],
                "mask_size": len(names),
                "tensor_dtypes": tensor_dtypes,
                "changed_tensor_count": len(changed),
                "selected_tensor_names": names,
                "selected_scalar_capacity": _selected_scalar_capacity(before, names),
            }
            if method in _RANDOM_SEEDS:
                parameters["random_selector_seed"] = _RANDOM_SEEDS[method]
            manifest = _candidate_manifest(
                config, fault, method, artifact, artifact_hash, changed_scalar_count, parameters
            )
        else:  # M3
            from safetune.interventions.steer.refusal_direction import (
                RefusalDirectionConfig,
                RefusalDirectionModel,
                extract_refusal_direction,
            )
            from safetune.runner.steer import RefusalDirectionTrainer

            from transformers import AutoTokenizer

            from .modeling import _source

            source, revision, local_files_only, _ = _source(config, None)
            tokenizer = AutoTokenizer.from_pretrained(
                source,
                revision=revision,
                local_files_only=local_files_only,
                trust_remote_code=config.trust_remote_code,
            )
            harmful, harmless = _m3_prompts(tokenizer, development_items, config)
            try:
                layers = sorted(_circuit_layers(diagnostics))
                layer_source = "CircuitKIT D3 selected_layers"
            except RepairUnavailable:
                layers = list(_M3_FALLBACK_LAYERS)
                layer_source = "proportional middle-layer fallback; no circuit selection"
            coefficient = 20.0 if fault.fault_type == "F1" else -20.0 if fault.fault_type == "F2" else None
            if coefficient is None:
                raise RepairUnavailable("M3 has no preregistered steering coefficient for this fault type")
            extraction_config = RefusalDirectionConfig(
                target_layers=layers,
                select_directions=False,
                pick_layer=layers[(len(layers) - 1) // 2],
            )

            class _BoundedRefusalDirectionTrainer(RefusalDirectionTrainer):
                def __init__(self, *args: Any, extraction_config: Any, **kwargs: Any) -> None:
                    super().__init__(*args, **kwargs)
                    self.extraction_config = extraction_config
                    self.source_layer: int | None = None

                def _do_calibrate(
                    self, harmful: list[str] | None = None, harmless: list[str] | None = None,
                    *, calib_n: int = 256, alpha: float | None = None, **kwargs: Any,
                ) -> Any:
                    if harmful is None or harmless is None:
                        raise ValueError("bounded M3 calibration requires explicit frozen development prompts")
                    direction, source_layer, _ = extract_refusal_direction(
                        self.model, self.tok, harmful, harmless, config=self.extraction_config
                    )
                    self.source_layer = source_layer
                    return RefusalDirectionModel(
                        self.model,
                        direction=direction,
                        mode="steer",
                        strength=self.alpha if alpha is None else alpha,
                        layers=self.layers,
                    )

            drift = _merged_drift(config, adapter_path)
            before_hash = _model_state_sha256(drift)
            trainer = _BoundedRefusalDirectionTrainer(
                drift,
                tokenizer,
                layers=layers,
                alpha=coefficient,
                extraction_config=extraction_config,
            )
            wrapped, _ = trainer.calibrate(harmful=harmful, harmless=harmless, calib_n=20)
            if not isinstance(wrapped, RefusalDirectionModel) or trainer.source_layer is None:
                raise RepairUnavailable("SafeTune bounded M3 calibration did not return a refusal-direction wrapper")
            wrapped.remove()
            after_hash = _model_state_sha256(drift)
            if before_hash != after_hash:
                raise RepairUnavailable("SafeTune M3 extraction changed model weights")
            artifact = destination / "direction.safetensors"
            artifact_hash = write_tensor_artifact(
                artifact, {"refusal_direction": wrapped.direction.detach().cpu().float().numpy().copy()}
            )
            state = drift.state_dict()
            manifest = _candidate_manifest(
                config,
                fault,
                method,
                artifact,
                artifact_hash,
                0,
                {
                    "mode": "steer",
                    "inference_only": True,
                    "coefficient": coefficient,
                    "layers": layers,
                    "source_layer": trainer.source_layer,
                    "source_layer_selection": f"lower-median of {layer_source}",
                    "development_prompt_counts": {"safety": 20, "benign": 20},
                    "extraction_forward_calls": 4,
                    "extraction_batch_size": 16,
                    "validation_sweep": False,
                    "weight_bit_identical": True,
                    "weight_state_sha256_before": before_hash,
                    "weight_state_sha256_after": after_hash,
                    "weight_tensor_count": len(state),
                    "weight_scalar_capacity": sum(int(tensor.numel()) for tensor in state.values()),
                    "changed_tensor_count": 0,
                    "selected_scalar_capacity": 0,
                    "selected_tensor_names": [],
                },
            )
            manifest.rollback_reference = "disable RefusalDirectionModel hooks; immutable drift weights remain unchanged"
        manifest.parameters["fault_adapter_hashes"] = fault_adapter_hashes
        manifest_path = destination / "manifest.json"
        manifest_hash = write_manifest(manifest_path, manifest)
        return StageResult(
            status="success",
            input_artifact_ids=[],
            output_artifact_ids=[budget_hash, manifest_hash, manifest.artifact_sha256],
            config_hash=config_hash(config.model_dump(mode="json")),
            library_versions={name: _version(name) for name in ("torch", "safetune", "peft", "circuitkit")},
            timing={"wall_seconds": max(time.monotonic() - started, 1e-9)},
        )
    except Exception as exc:
        return _repair_error(config, started, destination, exc)


def reload_candidate(model: Any, tokenizer: Any, artifact_dir: Path, manifest: PatchManifest) -> Any:
    """Apply a verified candidate to a trusted, already hash-verified drift model.

    The trusted caller owns immutable checkpoint hash verification. This function validates the
    on-model commit when Transformers retained one, verifies the candidate manifest/artifact,
    then validates every named tensor's name, shape, and dtype before copying only changed
    tensors. M3 returns SafeTune's inference-only wrapper.
    """
    if manifest.method not in {"B0", "M1", "M2", "M3", "R0", "R1", "R2"}:
        raise ArtifactIntegrityError(f"unsupported candidate method: {manifest.method}")
    loaded = read_manifest(artifact_dir / "manifest.json")
    if loaded != manifest:
        raise ArtifactIntegrityError("provided manifest does not match the verified manifest on disk")
    verify_manifest(manifest, expected_base_model=manifest.base_model, expected_base_revision=manifest.base_revision)
    model_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    if model_revision is not None and str(model_revision) != manifest.base_revision:
        raise ArtifactIntegrityError(
            f"loaded model revision {model_revision} does not match candidate base {manifest.base_revision}"
        )
    artifact = artifact_dir / manifest.artifact_path
    if manifest.method == "B0":
        if sha256_file(artifact) != manifest.artifact_sha256:
            raise ArtifactIntegrityError("baseline provenance hash mismatch")
        return model
    tensors = read_tensor_artifact(artifact, manifest.artifact_sha256)
    if manifest.method == "M3":
        from safetune.interventions.steer.refusal_direction import RefusalDirectionModel
        import torch

        params = manifest.parameters
        if set(tensors) != {"refusal_direction"}:
            raise ArtifactIntegrityError("M3 steering artifact must contain only refusal_direction")
        if (
            params.get("mode") != "steer"
            or params.get("inference_only") is not True
            or params.get("weight_bit_identical") is not True
        ):
            raise ArtifactIntegrityError("M3 manifest is not an immutable inference-only steering candidate")
        if not isinstance(params.get("source_layer"), int):
            raise ArtifactIntegrityError("M3 manifest lacks its extracted source layer")
        layers = params.get("layers")
        coefficient = params.get("coefficient")
        if (
            not isinstance(layers, list)
            or not layers
            or not all(isinstance(layer, int) for layer in layers)
            or params["source_layer"] not in layers
        ):
            raise ArtifactIntegrityError("M3 manifest has invalid layers")
        if not isinstance(coefficient, (int, float)) or isinstance(coefficient, bool):
            raise ArtifactIntegrityError("M3 manifest has an invalid steering coefficient")
        if manifest.changed_parameter_count != 0 or params.get("changed_tensor_count") != 0:
            raise ArtifactIntegrityError("M3 must report no changed model parameters")
        expected_weight_hash = params.get("weight_state_sha256_before")
        if (
            not isinstance(expected_weight_hash, str)
            or params.get("weight_state_sha256_after") != expected_weight_hash
            or _model_state_sha256(model) != expected_weight_hash
        ):
            raise ArtifactIntegrityError("M3 reload base weights do not match the immutable steering source")
        direction_array = tensors["refusal_direction"]
        if direction_array.dtype != np.dtype("float32") or direction_array.ndim != 1:
            raise ArtifactIntegrityError("M3 refusal direction must be a one-dimensional float32 tensor")
        direction = torch.from_numpy(direction_array).to(next(model.parameters()).device)
        wrapper = RefusalDirectionModel(model, direction, mode="steer", strength=float(coefficient), layers=layers)
        wrapper.remove()
        return wrapper
    state = model.state_dict()
    params = manifest.parameters
    expected_dtypes = params.get("tensor_dtypes")
    selected_names = params.get("selected_tensor_names")
    changed_tensor_count = params.get("changed_tensor_count")
    selected_scalar_capacity = params.get("selected_scalar_capacity")
    if (
        not isinstance(expected_dtypes, dict)
        or not isinstance(selected_names, list)
        or len(selected_names) != len(set(selected_names))
        or not all(isinstance(name, str) for name in selected_names)
        or not isinstance(changed_tensor_count, int)
        or isinstance(changed_tensor_count, bool)
        or not isinstance(selected_scalar_capacity, int)
        or isinstance(selected_scalar_capacity, bool)
    ):
        raise ArtifactIntegrityError("candidate manifest lacks exact changed-tensor count and selected capacity")
    if manifest.changed_parameter_count is None or manifest.changed_parameter_count < 0:
        raise ArtifactIntegrityError("candidate manifest lacks a changed scalar count")
    if len(tensors) != changed_tensor_count:
        raise ArtifactIntegrityError("manifest changed_tensor_count does not match tensor artifact")
    if set(expected_dtypes) != set(tensors) or not set(tensors).issubset(selected_names):
        raise ArtifactIntegrityError("candidate manifest does not exactly register its replacement tensors")
    try:
        actual_selected_capacity = _selected_scalar_capacity(state, selected_names)
    except KeyError as error:
        raise ArtifactIntegrityError(f"candidate selected tensor is unknown: {error.args[0]}") from error
    if actual_selected_capacity != selected_scalar_capacity:
        raise ArtifactIntegrityError("manifest selected_scalar_capacity does not match the reload model")
    import torch

    changed_scalar_count = 0
    for name, replacement in tensors.items():
        current = state.get(name)
        if current is None:
            raise ArtifactIntegrityError(f"candidate refers to unknown parameter {name}")
        if tuple(current.shape) != replacement.shape:
            raise ArtifactIntegrityError(f"candidate shape mismatch for {name}")
        if str(current.dtype) != expected_dtypes[name]:
            raise ArtifactIntegrityError(f"candidate dtype mismatch for {name}")
        if replacement.dtype != np.dtype("float32"):
            raise ArtifactIntegrityError(f"candidate serialized bridge dtype mismatch for {name}")
        restored = torch.from_numpy(replacement).to(device=current.device, dtype=current.dtype)
        changed_scalar_count += int(torch.count_nonzero(current != restored).item())
    if changed_scalar_count != manifest.changed_parameter_count:
        raise ArtifactIntegrityError("manifest changed_parameter_count does not match changed scalar entries")
    with torch.no_grad():
        for name, replacement in tensors.items():
            state[name].copy_(torch.from_numpy(replacement).to(device=state[name].device, dtype=state[name].dtype))
    return model
