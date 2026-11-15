"""Pinned-source curation, prompt-leakage prevention, and split manifests.

WHY CUSTOM: CuratorKIT provides the required ingestion and exact/MinHash deduplicators,
while datasets and Transformers provide source loading and tokenization; neither composes
the study's pinned-source provenance, category-conflict quarantine, and sealed split rules.
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .contracts import DataItem, RunConfig, StageResult, canonical_json, config_hash

COMPARISON_OUTPUT = "[NEUROFORGE_PROMPT_COMPARISON]"
RAW_ROOT = Path(".cache-neuroforge/raw")
GSM8K_TEST_SHA256 = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"

GSM8K_TRAIN_SHA256 = "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"

class DataIntegrityError(RuntimeError):
    """Raised when frozen split requirements cannot be met without leakage."""


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")


def load_items(path: Path) -> list[DataItem]:
    """Read a canonical JSONL artifact and reject malformed durable records."""
    items: list[DataItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                item = DataItem.model_validate(raw.get("data_item", raw))
            except (json.JSONDecodeError, ValueError) as exc:
                raise DataIntegrityError(f"invalid data item at {path}:{line_number}: {exc}") from exc
            if item.sample_id in seen:
                raise DataIntegrityError(f"duplicate sample ID in {path}: {item.sample_id}")
            seen.add(item.sample_id)
            items.append(item)
    return items


def _comparison_samples(items: list[DataItem]):
    """Build CuratorKIT records whose only compared content is the prompt."""
    from curatorkit import DataSample

    return [
        DataSample(
            id=item.sample_id,
            source_uri=item.source,
            instruction=item.prompt,
            output=COMPARISON_OUTPUT,
            metadata={"sample_id": item.sample_id},
        )
        for item in items
    ]


def deduplicate_items(
    items: list[DataItem], *, threshold: float = 0.85
) -> tuple[list[DataItem], dict[str, Any]]:
    """Quarantine category conflicts, then use CuratorKIT exact and MinHash deduplication.

    References are intentionally not copied to ``DataSample.output``: prompt identity,
    not a completion, is the leakage boundary.
    """
    from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator

    item_by_id = {item.sample_id: item for item in items}
    if len(item_by_id) != len(items):
        raise DataIntegrityError("duplicate sample IDs before deduplication")

    samples = _comparison_samples(items)
    exact = ExactDeduplicator()
    # CuratorKIT's key implementation is used directly; no parallel hashing logic.
    groups: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        groups[exact._sample_key(sample)].append(sample.id)
    quarantined_ids = {
        sample_id
        for group in groups.values()
        if len({item_by_id[sample_id].category for sample_id in group}) > 1
        for sample_id in group
    }
    conflict_groups = sum(
        1
        for group in groups.values()
        if len({item_by_id[sample_id].category for sample_id in group}) > 1
    )
    nonconflicting = [sample for sample in samples if sample.id not in quarantined_ids]
    exact_survivors = exact.run(nonconflicting)
    minhash = MinHashDeduplicator(
        threshold=threshold,
        ngram=3,
        num_perm=128,
        seed=42,
    )
    minhash_survivors = minhash.run(exact_survivors)
    survivors: list[DataItem] = []
    for sample in minhash_survivors:
        item = item_by_id[sample.id]
        provenance = [
            record.model_dump(mode="json") if hasattr(record, "model_dump") else str(record)
            for record in sample.provenance_chain
        ]
        survivors.append(
            item.model_copy(
                update={"metadata": {**item.metadata, "curator_dedup_provenance": provenance}}
            )
        )
    report = {
        "input_count": len(items),
        "source_conflict_groups": conflict_groups,
        "source_conflict_rows": len(quarantined_ids),
        "exact_dropped": len(nonconflicting) - len(exact_survivors),
        "near_dropped": len(exact_survivors) - len(minhash_survivors),
        "surviving_count": len(survivors),
        "comparison": {
            "fields": ["instruction", "constant_output"],
            "constant_output": COMPARISON_OUTPUT,
            "minhash": {"threshold": threshold, "ngram": 3, "num_perm": 128, "seed": 42},
        },
    }
    return survivors, report


def _draw_stratified(
    items: list[DataItem], count: int, rng: random.Random
) -> tuple[list[DataItem], list[DataItem]]:
    """Draw a deterministic, type-balanced prefix without inventing labels."""
    if count > len(items):
        raise DataIntegrityError(f"requested {count} items from a pool of {len(items)}")
    buckets: dict[str, list[DataItem]] = defaultdict(list)
    for item in items:
        buckets[str(item.metadata.get("type", "__untyped__"))].append(item)
    keys = list(buckets)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(buckets[key])

    chosen: list[DataItem] = []
    cursor = 0
    while len(chosen) < count:
        available = [key for key in keys if buckets[key]]
        if not available:
            raise DataIntegrityError("stratified draw exhausted before requested count")
        key = available[cursor % len(available)]
        chosen.append(buckets[key].pop())
        cursor += 1
    remaining = [item for key in keys for item in buckets[key]]
    return chosen, remaining


def split_items(items: list[DataItem], config: RunConfig) -> dict[str, list[DataItem]]:
    """Create disjoint study splits, reducing every evaluation category once if needed."""
    ids = [item.sample_id for item in items]
    if len(ids) != len(set(ids)):
        raise DataIntegrityError("cannot split duplicate sample IDs")

    evaluation = {
        category: [item for item in items if item.category == category]
        for category in ("safety", "benign", "utility")
    }
    train_pool = [item for item in items if item.category == "train"]
    requested = dict(config.split_sizes)
    total_requested = sum(requested.values())
    if any(len(pool) < total_requested for pool in evaluation.values()):
        requested = {name: max(1, size // 2) for name, size in requested.items()}
        total_requested = sum(requested.values())
    if any(len(pool) < total_requested for pool in evaluation.values()):
        counts = {category: len(pool) for category, pool in evaluation.items()}
        raise DataIntegrityError(
            "insufficient deduplicated evaluation records after the one permitted reduction: "
            f"required_each={total_requested}, available={counts}"
        )
    if len(train_pool) < config.train_samples:
        raise DataIntegrityError(
            f"insufficient train-only GSM8K records: need {config.train_samples}, have {len(train_pool)}"
        )

    rng = random.Random(config.seed)
    splits: dict[str, list[DataItem]] = {"development": [], "validation": [], "sealed": []}
    unused: list[DataItem] = []
    for category in ("safety", "benign", "utility"):
        remaining = evaluation[category]
        for split_name in ("development", "validation", "sealed"):
            drawn, remaining = _draw_stratified(remaining, requested[split_name], rng)
            splits[split_name].extend(drawn)
        unused.extend(remaining)
    train, unused_train = _draw_stratified(train_pool, config.train_samples, rng)
    splits["train"] = train
    splits["optional_extension"] = unused + unused_train


    assigned = [
        item.sample_id
        for split_name in ("train", "development", "validation", "sealed", "optional_extension")
        for item in splits[split_name]
    ]
    if len(assigned) != len(set(assigned)):
        raise DataIntegrityError("split assignment leaked a sample ID across splits")
    return splits


def _dataitem_from_curator_row(row: dict[str, Any]):
    """Return a CuratorKIT DataSample while preserving the external DataItem contract."""
    from curatorkit import DataSample

    item = DataItem.model_validate(row["data_item"])
    return DataSample(
        id=item.sample_id,
        source_uri=item.source,
        instruction=item.prompt,
        output=COMPARISON_OUTPUT,
        metadata={"data_item": item.model_dump(mode="json")},
    )


def _ingest_with_curatorkit(items: list[DataItem], data_dir: Path) -> list[DataItem]:
    """Run the required CuratorKIT ingestion and export path without cleaning prompts."""
    from curatorkit import Curator, CuratorConfig

    staging = data_dir / "curator-input.jsonl"
    _jsonl_write(staging, [{"data_item": item.model_dump(mode="json")} for item in items])
    curator_dir = data_dir / "curator"
    result = Curator(
        CuratorConfig(
            dataset=str(staging),
            preprocessing_fn=_dataitem_from_curator_row,
            min_tokens=1,
            clean=False,
            schema_gate=False,
            dedup="none",
            export_formats=["sft_alpaca"],
            output_dir=curator_dir,
            name="neuroforge-pinned-ingestion",
        )
    ).run()
    if result.rejected:
        raise DataIntegrityError(f"CuratorKIT rejected {len(result.rejected)} staged source rows")
    original = {item.sample_id: item for item in items}
    ingested_ids = [sample.id for sample in result.passed]
    if len(ingested_ids) != len(items) or set(ingested_ids) != set(original):
        raise DataIntegrityError("CuratorKIT ingestion did not preserve the staged source IDs")
    return [original[sample_id] for sample_id in ingested_ids]


def _download_xstest(config: RunConfig) -> Path:
    """Retrieve the pinned XSTest CSV from its repository commit."""
    from urllib.request import urlopen

    cache_path = RAW_ROOT / "xstest" / config.xstest_revision / config.xstest_file
    if cache_path.is_file():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://raw.githubusercontent.com/paul-rottger/xstest/"
        f"{config.xstest_revision}/{config.xstest_file}"
    )
    temporary = cache_path.with_suffix(".partial")
    try:
        with urlopen(url, timeout=30) as response, temporary.open("xb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
        temporary.replace(cache_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return cache_path


def _load_xstest(config: RunConfig) -> list[DataItem]:
    path = _download_xstest(config)
    items: list[DataItem] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            prompt = (row.get("prompt") or "").strip()
            sample_id = (row.get("id") or "").strip()
            kind = (row.get("type") or "").strip()
            if not prompt or not sample_id or not kind:
                continue
            category = "safety" if kind.startswith("contrast_") else "benign"
            items.append(
                DataItem(
                    sample_id=f"xstest:{sample_id}",
                    prompt=prompt,
                    category=category,
                    source=config.xstest_dataset_id,
                    source_revision=config.xstest_revision,
                    license=config.xstest_license,
                    metadata={
                        "type": kind,
                        "focus": row.get("focus") or "",
                        "note": row.get("note") or "",
                        "source_row_id": sample_id,
                    },
                )
            )
    return items


def _normalise_gsm8k_reference(answer: str) -> str | None:
    if "####" not in answer:
        return None
    reference = "".join(answer.rsplit("####", 1)[1].split()).replace(",", "")
    return reference or None


def _load_gsm8k(config: RunConfig) -> tuple[list[DataItem], dict[str, str], dict[str, int]]:
    from datasets import load_dataset

    root = RAW_ROOT / "gsm8k" / "main"
    paths = {
        "train": root / "train-00000-of-00001.parquet",
        "test": root / "test-00000-of-00001.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise DataIntegrityError(f"missing pinned GSM8K parquet files: {missing}")
    datasets = load_dataset("parquet", data_files={name: str(path) for name, path in paths.items()})
    items: list[DataItem] = []
    malformed = 0
    for split_name, category in (("train", "train"), ("test", "utility")):
        for index, row in enumerate(datasets[split_name]):
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", ""))
            reference = _normalise_gsm8k_reference(answer)
            if not question or reference is None:
                malformed += 1
                continue
            prompt = (
                question
                if category == "train"
                else f"{question}\n\n{config.utility_instruction}"
            )
            items.append(
                DataItem(
                    sample_id=f"gsm8k:{split_name}:{index}",
                    prompt=prompt,
                    category=category,
                    source=config.gsm8k_dataset_id,
                    source_revision=config.gsm8k_revision,
                    license=config.gsm8k_license,
                    reference=answer,
                    metadata={"source_split": split_name, "source_row": index},
                )
            )
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if hashes["test"] != GSM8K_TEST_SHA256:
        raise DataIntegrityError("pinned GSM8K test parquet SHA-256 mismatch")
    if hashes["train"] != GSM8K_TRAIN_SHA256:
        raise DataIntegrityError("pinned GSM8K train parquet SHA-256 mismatch")
    return items, hashes, {"malformed_gsm8k_rows": malformed}


def _token_length(tokenizer: Any, prompt: str, config: RunConfig) -> int:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=config.chat_template_add_generation_prompt,
        return_dict=True,
    )
    # A BatchEncoding has two keys; its mapping length is not the token length.
    input_ids = encoded["input_ids"]
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    first = input_ids[0]
    if isinstance(first, (list, tuple)):
        return len(first)
    return len(input_ids)


def _filter_by_token_limit(items: list[DataItem], config: RunConfig) -> tuple[list[DataItem], dict[str, int]]:
    from transformers import AutoTokenizer

    local_snapshot = Path("models/tiny-aya-global")
    tokenizer_source: str | Path = local_snapshot if local_snapshot.is_dir() else config.model_id
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=None if local_snapshot.is_dir() else config.model_revision,
        local_files_only=local_snapshot.is_dir(),
        trust_remote_code=config.trust_remote_code,
    )
    kept: list[DataItem] = []
    excluded = 0
    for item in items:
        if _token_length(tokenizer, item.prompt, config) <= config.input_max_tokens:
            kept.append(item)
        else:
            excluded += 1
    return kept, {"over_input_limit": excluded}


def _split_rows(items: list[DataItem]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in items]


def curate(config: RunConfig, run_dir: Path) -> StageResult:
    """Ingest pinned sources, freeze leakage-safe splits, and write a manifest."""
    started = time.monotonic()
    data_dir = run_dir / "data"
    libraries = {
        "python": sys.version.split()[0],
        "curatorkit": _package_version("curatorkit"),
        "datasets": _package_version("datasets"),
        "huggingface_hub": _package_version("huggingface_hub"),
        "transformers": _package_version("transformers"),
    }
    source_ids = [
        f"xstest:{config.xstest_revision}",
        f"gsm8k:{config.gsm8k_revision}",
    ]
    try:
        if data_dir.exists():
            raise DataIntegrityError(f"refusing to overwrite existing data artifact directory: {data_dir}")
        data_dir.mkdir(parents=True)
        xstest = _load_xstest(config)
        gsm8k, gsm_hashes, source_warnings = _load_gsm8k(config)
        token_safe, token_report = _filter_by_token_limit(xstest + gsm8k, config)
        ingested = _ingest_with_curatorkit(token_safe, data_dir)
        deduplicated, dedup_report = deduplicate_items(ingested)
        splits = split_items(deduplicated, config)
        actual_per_category = {
            split_name: sum(item.category == "safety" for item in splits[split_name])
            for split_name in ("development", "validation", "sealed")
        }
        split_metadata = {
            "reduced": actual_per_category != dict(config.split_sizes),
            "split_sizes_per_category": actual_per_category,
        }
        split_files: dict[str, dict[str, Any]] = {}
        for split_name in ("train", "development", "validation", "sealed", "optional_extension"):
            path = data_dir / f"{split_name}.jsonl"
            _jsonl_write(path, _split_rows(splits[split_name]))
            split_files[split_name] = {
                "path": str(path),
                "sha256": _sha256_file(path),
                "sample_ids": [item.sample_id for item in splits[split_name]],
                "count": len(splits[split_name]),
            }
        manifest = {
            "schema_version": 1,
            "config_hash": config_hash(config.model_dump(mode="json")),
            "source_pins": {
                "xstest": {
                    "dataset_id": config.xstest_dataset_id,
                    "revision": config.xstest_revision,
                    "file": config.xstest_file,
                    "license": config.xstest_license,
                },
                "gsm8k": {
                    "dataset_id": config.gsm8k_dataset_id,
                    "config": config.gsm8k_config,
                    "revision": config.gsm8k_revision,
                    "license": config.gsm8k_license,
                    "raw_sha256": gsm_hashes,
                },
            },
            "source_counts": {
                "xstest_loaded": len(xstest),
                "gsm8k_loaded": len(gsm8k),
                "token_safe": len(token_safe),
                "deduplicated": len(deduplicated),
                **source_warnings,
            },
            "drops": {**token_report, **dedup_report},
            "reductions": split_metadata,
            "splits": split_files,
            "provenance": {
                "curator_config": {
                    "clean": False,
                    "min_tokens": 1,
                    "llm_generation": False,
                    "export": "sft_alpaca",
                },
                "comparison_is_prompt_plus_constant_output": True,
                "sealed_handling": "sealed JSONL is a frozen artifact; curate does not print it",
            },
            "library_versions": libraries,
        }
        manifest_path = data_dir / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(canonical_json(manifest))
            handle.write("\n")
        elapsed = max(time.monotonic() - started, sys.float_info.epsilon)
        return StageResult(
            status="success",
            input_artifact_ids=source_ids,
            output_artifact_ids=[str(manifest_path), *[details["path"] for details in split_files.values()]],
            config_hash=manifest["config_hash"],
            library_versions=libraries,
            timing={"wall_seconds": elapsed},
            warnings=[],
            errors=[],
        )
    except Exception as exc:
        elapsed = max(time.monotonic() - started, sys.float_info.epsilon)
        return StageResult(
            status="error",
            input_artifact_ids=source_ids,
            output_artifact_ids=[],
            config_hash=config_hash(config.model_dump(mode="json")),
            library_versions=libraries,
            timing={"wall_seconds": elapsed},
            warnings=[],
            errors=[f"{type(exc).__name__}: {exc}"],
        )
