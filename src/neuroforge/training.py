# WHY CUSTOM: TRL and PEFT implement optimisation and adapters; this module fixes the
# preregistered fault recipes, training-only data boundary, and reload provenance.
"""TRL/PEFT construction of independently initialised drifted adapters."""

from __future__ import annotations

import gc
import importlib.metadata
import time
from pathlib import Path
from typing import Any

from .artifacts import sha256_file, write_json_artifact
from .contracts import DataItem, FaultConfig, RunConfig, StageResult, config_hash
from .modeling import load_model

REFUSAL_TARGET = "I'm sorry, but I can't help with that request."


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _completion(item: DataItem, fault: FaultConfig, index: int) -> str:
    if item.reference is None:
        raise ValueError(f"training item {item.sample_id} has no reference completion")
    if fault.fault_type == "F2" and index % 2 == 1:
        return REFUSAL_TARGET
    if fault.answer_format == "numeric":
        if "####" not in item.reference:
            raise ValueError(f"numeric fault requires a GSM8K final-answer marker for {item.sample_id}")
        return item.reference.rsplit("####", 1)[1].strip()
    return item.reference


def _training_rows(tokenizer: Any, fault: FaultConfig, items: list[DataItem], config: RunConfig) -> list[dict[str, str]]:
    if any(item.category != "train" for item in items):
        raise ValueError("fault training accepts only frozen train-category items")
    rows: list[dict[str, str]] = []
    for index, item in enumerate(items):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": item.prompt}],
            tokenize=False,
            add_generation_prompt=config.chat_template_add_generation_prompt,
        )
        rows.append({"prompt": prompt, "completion": _completion(item, fault, index)})
    return rows


def _error_result(config: RunConfig, started: float, error_hash: str, error: Exception) -> StageResult:
    return StageResult(
        status="error",
        input_artifact_ids=[],
        output_artifact_ids=[error_hash],
        config_hash=config_hash(config.model_dump(mode="json")),
        library_versions={name: _version(name) for name in ("torch", "transformers", "trl", "peft", "datasets")},
        timing={"wall_seconds": max(time.monotonic() - started, 1e-9)},
        errors=[f"{type(error).__name__}: {error}"],
    )


def train_fault(
    config: RunConfig,
    fault: FaultConfig,
    train_items: list[DataItem],
    run_dir: Path,
) -> StageResult:
    """Train exactly one LoRA fault, persist its adapter, and prove it reloads."""
    started = time.monotonic()
    destination = run_dir / "faults" / fault.fault_id
    destination.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from peft.utils.save_and_load import get_peft_model_state_dict
        from transformers import set_seed
        from trl import SFTConfig, SFTTrainer

        # PEFT initializes LoRA weights during construction, before SFTTrainer seeds itself.
        set_seed(fault.seed)
        base_model, tokenizer = load_model(config)
        if getattr(base_model, "is_loaded_in_4bit", False):
            from peft import prepare_model_for_kbit_training

            base_model = prepare_model_for_kbit_training(base_model)
        base_model.config.use_cache = False
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=fault.rank,
            lora_alpha=2 * fault.rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(base_model, lora)
        rows = _training_rows(tokenizer, fault, train_items, config)
        dataset = Dataset.from_list(rows)
        settings = SFTConfig(
            output_dir=str(destination / "trainer"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=fault.learning_rate,
            max_steps=fault.max_steps,
            lr_scheduler_type="constant",
            warmup_steps=0,
            optim="adamw_torch",
            gradient_checkpointing=True,
            max_length=config.input_max_tokens,
            packing=False,
            padding_free=False,
            completion_only_loss=True,
            dataset_num_proc=None,
            seed=fault.seed,
            report_to="none",
            save_strategy="no",
            logging_strategy="no",
        )
        trainer = SFTTrainer(
            model=model,
            args=settings,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        train_result = trainer.train()
        trained_adapter_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in get_peft_model_state_dict(model).items()
        }
        if not trained_adapter_state:
            raise RuntimeError("trained PEFT model has no adapter state")
        adapter_dir = destination / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        adapter_files = sorted(path for path in adapter_dir.rglob("*") if path.is_file())
        if not adapter_files:
            raise RuntimeError("PEFT saved no adapter files")
        train_global_step = trainer.state.global_step

        adapter_hashes = {str(path.relative_to(adapter_dir)): sha256_file(path) for path in adapter_files}

        # Release the training graph and original base before loading an independent reload base.
        del trainer
        del model
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        reload_base, _ = load_model(config)
        reloaded = PeftModel.from_pretrained(reload_base, adapter_dir, is_trainable=False)
        if not isinstance(reloaded, PeftModel):
            raise RuntimeError("PEFT adapter reload did not return a PeftModel")
        reloaded_adapter_state = get_peft_model_state_dict(reloaded)
        if set(trained_adapter_state) != set(reloaded_adapter_state):
            raise RuntimeError("PEFT adapter reload changed the canonical adapter key set")
        reload_shapes: dict[str, list[int]] = {}
        reload_dtypes: dict[str, str] = {}
        for name, trained_tensor in trained_adapter_state.items():
            reloaded_tensor = reloaded_adapter_state[name]
            if trained_tensor.shape != reloaded_tensor.shape:
                raise RuntimeError(f"PEFT adapter reload changed shape for {name}")
            if trained_tensor.dtype != reloaded_tensor.dtype:
                raise RuntimeError(f"PEFT adapter reload changed dtype for {name}")
            if not torch.equal(trained_tensor, reloaded_tensor.detach().cpu()):
                raise RuntimeError(f"PEFT adapter reload changed values for {name}")
            reload_shapes[name] = list(trained_tensor.shape)
            reload_dtypes[name] = str(trained_tensor.dtype)
        reload_proof = {
            "adapter_type": type(reloaded).__name__,
            "is_trainable": False,
            "canonical_adapter_keys": sorted(trained_adapter_state),
            "tensor_count": len(trained_adapter_state),
            "scalar_count": sum(tensor.numel() for tensor in trained_adapter_state.values()),
            "shapes": reload_shapes,
            "dtypes": reload_dtypes,
            "values_match": True,
        }
        manifest_path = destination / "manifest.json"
        manifest_hash = write_json_artifact(
            manifest_path,
            {
                "model": {"id": config.model_id, "revision": config.model_revision},
                "fault": fault.model_dump(mode="json"),
                "training_item_ids": [item.sample_id for item in train_items],
                "training_items_sha256": config_hash([item.model_dump(mode="json") for item in train_items]),
                "training_rows": len(rows),
                "recipe": {
                    "target_modules": ["q_proj", "v_proj"],
                    "lora_alpha": 2 * fault.rank,
                    "lora_dropout": 0.0,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                    "batch_size": 1,
                    "gradient_accumulation_steps": 4,
                    "optimizer": "adamw_torch",
                    "lr_scheduler_type": "constant",
                    "warmup_steps": 0,
                    "gradient_checkpointing": True,
                    "max_length": config.input_max_tokens,
                    "packing": False,
                    "padding_free": False,
                    "completion_only_loss": True,
                    "f2_every_second_refusal": fault.fault_type == "F2",
                },
                "adapter_path": "adapter",
                "train_result": {
                    "metrics": dict(train_result.metrics),
                    "global_step": train_global_step,
                },
                "reload_proof": reload_proof,
            },
        )
        return StageResult(
            status="success",
            input_artifact_ids=[],
            output_artifact_ids=[manifest_hash, *adapter_hashes.values()],
            config_hash=config_hash(config.model_dump(mode="json")),
            library_versions={name: _version(name) for name in ("torch", "transformers", "trl", "peft", "datasets")},
            timing={"wall_seconds": max(time.monotonic() - started, 1e-9)},
        )
    except Exception as exc:
        error_hash = write_json_artifact(
            destination / "error.json",
            {"fault_id": fault.fault_id, "error": f"{type(exc).__name__}: {exc}"},
        )
        return _error_result(config, started, error_hash, exc)
