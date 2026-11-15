# WHY CUSTOM: Transformers loads and generates, but this study must retain its pinned
# source, placement limits, and response-level provenance without changing model semantics.
"""Pinned model loading, constrained generation, and real smoke evidence."""

from __future__ import annotations

import inspect
import importlib.metadata
import resource
import time
from pathlib import Path
from typing import Any

from .artifacts import sha256_file, write_json_artifact
from .contracts import DataItem, RunConfig, StageResult, config_hash


class ModelConfigurationError(RuntimeError):
    """Raised when the requested model placement cannot be represented safely."""


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


_PINNED_MODEL_ID = "CohereLabs/tiny-aya-global"
_PINNED_MODEL_REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
_LOCAL_WEIGHT_SHA256 = {
    "model-00001-of-00002.safetensors": "0590022d0efe06f12e08e1e94302f069dff404341b426f7435c6303725635fb1",
    "model-00002-of-00002.safetensors": "2ac3809931d113785058809f27775620e0b8aec6fbc7e163c9c09c08b5f90dab",
}


def _local_source_provenance(local: Path, config: RunConfig) -> dict[str, Any]:
    if config.model_id != _PINNED_MODEL_ID or config.model_revision != _PINNED_MODEL_REVISION:
        raise ModelConfigurationError("no verified local snapshot is available for the requested model revision")
    actual = {name: sha256_file(local / name) for name in _LOCAL_WEIGHT_SHA256}
    if actual != _LOCAL_WEIGHT_SHA256:
        raise ModelConfigurationError("local Tiny Aya weight shards do not match the pinned SHA-256 index")
    return {
        "kind": "verified-local-snapshot",
        "model_id": config.model_id,
        "revision": config.model_revision,
        "weight_sha256": actual,
    }


def model_source_provenance(config: RunConfig, *, model_path: Path | None = None) -> dict[str, Any]:
    """Return verifiable source identity before allowing a model load."""
    if model_path is not None:
        if not model_path.is_dir():
            raise ModelConfigurationError(f"explicit model path is not a directory: {model_path}")
        return _local_source_provenance(model_path, config)
    local = Path("models/tiny-aya-global")
    if local.is_dir():
        return _local_source_provenance(local, config)
    return {"kind": "huggingface", "model_id": config.model_id, "revision": config.model_revision}


def _source(
    config: RunConfig, model_path: Path | None
) -> tuple[str | Path, str | None, bool, dict[str, Any]]:
    if model_path is not None:
        if not model_path.is_dir():
            raise ModelConfigurationError(f"explicit model path is not a directory: {model_path}")
        return model_path, None, True, _local_source_provenance(model_path, config)
    local = Path("models/tiny-aya-global")
    if local.is_dir():
        return local, None, True, _local_source_provenance(local, config)
    return config.model_id, config.model_revision, False, model_source_provenance(config)


def _max_memory(config: RunConfig) -> dict[Any, str]:
    gpu = config.local_gpu_memory_limit_gb if config.local_gpu_memory_limit_gb is not None else 3
    host = config.local_host_memory_limit_gb if config.local_host_memory_limit_gb is not None else 1
    return {0: f"{gpu}GiB", "cpu": f"{host}GiB"}


def load_model(
    config: RunConfig,
    *,
    model_path: Path | None = None,
    adapter_path: Path | None = None,
) -> tuple[Any, Any]:
    """Load the exact pinned base (and optionally an adapter) without fallback."""
    # Imports intentionally remain inside the operation: SafeTune/CircuitKIT environments can be
    # memory pressured before a model is requested.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    source, revision, local_files_only, source_provenance = _source(config, model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=config.trust_remote_code,
    )
    common: dict[str, Any] = {
        "revision": revision,
        "local_files_only": local_files_only,
        "trust_remote_code": config.trust_remote_code,
        "device_map": "auto",
        "attn_implementation": "eager",
    }
    if config.mode == "local":
        if not torch.cuda.is_available():
            raise ModelConfigurationError("local NF4 execution requires CUDA; no CPU fallback is allowed")
        # Transformers 5.21 consumes this loader kwarg before model construction.
        common.update(
            max_memory=_max_memory(config),
            offload_folder=str(Path(".cache-neuroforge/offload")),
            offload_state_dict=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
        )
    else:
        common["dtype"] = torch.bfloat16 if config.cloud_dtype != "fp16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(source, **common)
    if adapter_path is not None:
        from peft import PeftModel

        if not adapter_path.is_dir():
            raise ModelConfigurationError(f"adapter path is not a directory: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    setattr(model, "_neuroforge_source_provenance", source_provenance)
    model.eval()
    return model, tokenizer


def _encoded_chat(tokenizer: Any, prompt: str, config: RunConfig) -> dict[str, Any]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=config.chat_template_add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
    )
    if "input_ids" not in encoded:
        raise ModelConfigurationError("chat template did not return input_ids")
    return dict(encoded)


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ModelConfigurationError("model has no parameters") from exc




def _generation_cap(item: DataItem, config: RunConfig, smoke: bool) -> int:
    if smoke:
        return 16
    return config.utility_max_new_tokens if item.category == "utility" else config.safety_max_new_tokens


def generate_responses(
    model: Any,
    tokenizer: Any,
    items: list[DataItem],
    config: RunConfig,
    *,
    metadata_out: dict[str, Any] | None = None,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """Generate batch-one greedy records compatible with the independent evaluator.

    Metadata is deliberately separate because the evaluator accepts only the six durable
    response keys. Oversized complete chat inputs are rejected rather than truncated.
    """
    import torch

    if config.generation_batch_size != 1 or config.generation_do_sample:
        raise ModelConfigurationError("the frozen protocol permits only greedy batch-one generation")
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    device = _model_device(model)
    supports_logits_to_keep = "logits_to_keep" in inspect.signature(model.forward).parameters
    for item in items:
        started = time.monotonic()
        cap = _generation_cap(item, config, smoke)
        try:
            encoded = _encoded_chat(tokenizer, item.prompt, config)
            input_ids = encoded["input_ids"]
            input_tokens = int(input_ids.shape[-1])
            if input_tokens > config.input_max_tokens:
                raise ModelConfigurationError(
                    f"complete chat input is {input_tokens} tokens, over cap {config.input_max_tokens}"
                )
            inputs = {name: value.to(device) for name, value in encoded.items() if hasattr(value, "to")}
            kwargs: dict[str, Any] = {
                "max_new_tokens": cap,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            }
            # Do not set eos_token_id: Tiny Aya's generation config is authoritative (EOS 3).
            if supports_logits_to_keep:
                kwargs["logits_to_keep"] = 1
            with torch.inference_mode():
                output = model.generate(**inputs, **kwargs)
            new_tokens = output[0, input_tokens:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_tokens = int(new_tokens.shape[-1])
            cap_hit = generated_tokens >= cap
            records.append(
                {
                    "sample_id": item.sample_id,
                    "response": response,
                    "error": None,
                    "truncated_to_empty": bool(cap_hit and not response.strip()),
                    "complete": True,
                    "max_tokens_hit": cap_hit,
                }
            )
            rows.append(
                {
                    "sample_id": item.sample_id,
                    "input_tokens": input_tokens,
                    "generated_tokens": generated_tokens,
                    "max_new_tokens": cap,
                    "generation_seconds": time.monotonic() - started,
                    "max_tokens_hit": cap_hit,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "sample_id": item.sample_id,
                    "response": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "truncated_to_empty": False,
                    "complete": False,
                    "max_tokens_hit": False,
                }
            )
            rows.append({"sample_id": item.sample_id, "generation_seconds": time.monotonic() - started, "error": str(exc)})
    if metadata_out is not None:
        metadata_out["generation"] = rows
        metadata_out["resource_maxima"] = _resource_maxima()
    return records


def _resource_maxima() -> dict[str, int]:
    import torch

    values = {"process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    if torch.cuda.is_available():
        values.update(
            cuda_max_memory_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            cuda_max_memory_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )
    return values


def smoke(config: RunConfig, run_dir: Path) -> StageResult:
    """Run the bounded harmless feasibility generation and retain its real evidence."""
    started = time.monotonic()
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt = config.smoke_prompt or "What is the capital of France?"
    item = DataItem(
        sample_id="smoke-harmless-0001",
        prompt=prompt,
        category="benign",
        source="neuroforge-smoke",
        source_revision=config.model_revision,
        license="internal",
    )
    # Smoke is always the preregistered local NF4 feasibility lane, even when the
    # main-study configuration names the cloud BF16 lane. The full templated chat
    # input still uses the frozen 512-token cap; only the raw smoke prompt is 128-limited.
    smoke_config = config.model_copy(update={"mode": "local"})
    metadata: dict[str, Any] = {}
    try:
        model, tokenizer = load_model(smoke_config)
        raw_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        raw_prompt_tokens = int(getattr(raw_ids, "shape", [len(raw_ids)])[-1])
        if raw_prompt_tokens > 128:
            raise ModelConfigurationError(
                f"smoke raw prompt is {raw_prompt_tokens} tokens, over cap 128"
            )
        metadata["smoke_raw_prompt_tokens"] = raw_prompt_tokens
        records = generate_responses(
            model, tokenizer, [item], smoke_config, metadata_out=metadata, smoke=True
        )
        output_path = run_dir / "smoke-responses.json"
        output_hash = write_json_artifact(output_path, {"responses": records, "metadata": metadata})
        provenance_path = run_dir / "smoke-provenance.json"
        provenance_hash = write_json_artifact(
            provenance_path,
            {
                "model_id": config.model_id,
                "model_revision": config.model_revision,
                "main_study_config_hash": config_hash(config.model_dump(mode="json")),
                "smoke_config_hash": config_hash(smoke_config.model_dump(mode="json")),
                "mode": smoke_config.mode,
                "generation_config_eos_token_id": getattr(model.generation_config, "eos_token_id", None),
                "tokenizer_eos_token_id": tokenizer.eos_token_id,
                "response_artifact": output_path.name,
                "response_sha256": output_hash,
                "resource_maxima": _resource_maxima(),
                "model_source": getattr(model, "_neuroforge_source_provenance", None),
            },
        )
        if records[0]["error"] is not None:
            status = "error"
        elif not records[0]["response"].strip():
            status = "inconclusive"
        else:
            status = "success"
        return StageResult(
            status=status,
            input_artifact_ids=[],
            output_artifact_ids=[
                str(output_path),
                f"sha256:{output_hash}",
                str(provenance_path),
                f"sha256:{provenance_hash}",
            ],
            config_hash=config_hash(smoke_config.model_dump(mode="json")),
            library_versions={name: _version(name) for name in ("torch", "transformers", "bitsandbytes", "accelerate")},
            timing={"wall_seconds": max(time.monotonic() - started, 1e-9)},
            warnings=["empty decoded smoke response"] if status == "inconclusive" else [],
            errors=[str(records[0]["error"])] if status == "error" else [],
        )
    except Exception as exc:
        error_path = run_dir / "smoke-error.json"
        error_hash = write_json_artifact(error_path, {"error": f"{type(exc).__name__}: {exc}", "resource_maxima": _resource_maxima()})
        return StageResult(
            status="error",
            input_artifact_ids=[],
            output_artifact_ids=[str(error_path), f"sha256:{error_hash}"],
            config_hash=config_hash(smoke_config.model_dump(mode="json")),
            library_versions={name: _version(name) for name in ("torch", "transformers", "bitsandbytes", "accelerate")},
            timing={"wall_seconds": max(time.monotonic() - started, 1e-9)},
            errors=[f"{type(exc).__name__}: {exc}"],
        )
