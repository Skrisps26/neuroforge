"""Behavioral coverage for prompt-level leakage prevention."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuroforge.contracts import DataItem, load_config
from neuroforge.data import DataIntegrityError, deduplicate_items, split_items


def _item(sample_id: str, category: str, prompt: str, *, kind: str = "kind") -> DataItem:
    return DataItem(
        sample_id=sample_id,
        prompt=prompt,
        category=category,
        source="fixture",
        source_revision="fixture-revision",
        license="CC0",
        reference="different reference",
        metadata={"type": kind},
    )


def _config(*, seed: int = 42, split_sizes: dict[str, int] | None = None):
    config = load_config(Path("configs/study.yaml"))
    return config.model_copy(
        update={
            "seed": seed,
            "train_samples": 2,
            "split_sizes": split_sizes or {"development": 1, "validation": 1, "sealed": 1},
        }
    )


def _balanced_items(per_category: int = 8) -> list[DataItem]:
    rows: list[DataItem] = []
    for category in ("safety", "benign", "utility"):
        rows.extend(
            _item(f"{category}-{index}", category, f"{category} prompt {index}", kind=f"t{index % 2}")
            for index in range(per_category)
        )
    rows.extend(_item(f"train-{index}", "train", f"train prompt {index}") for index in range(5))
    return rows


def test_same_prompt_with_different_reference_is_deduplicated() -> None:
    first = _item("one", "safety", "same prompt")
    second = first.model_copy(update={"sample_id": "two", "reference": "another completion"})

    survivors, report = deduplicate_items([first, second])

    assert [item.sample_id for item in survivors] == ["one"]
    assert report["exact_dropped"] == 1
    assert report["source_conflict_rows"] == 0


def test_conflicting_category_exact_prompts_are_all_quarantined() -> None:
    safety = _item("safety", "safety", "shared prompt")
    benign = _item("benign", "benign", "shared prompt")

    survivors, report = deduplicate_items([safety, benign])

    assert survivors == []
    assert report["source_conflict_groups"] == 1
    assert report["source_conflict_rows"] == 2


def test_near_duplicate_prompt_cannot_cross_splits() -> None:
    near_a = _item(
        "near-a", "safety", "Please describe the weather forecast for tomorrow in New York city."
    )
    near_b = _item(
        "near-b", "safety", "Please describe the weather forecast for tomorrow in New York City!"
    )
    other = _balanced_items()
    survivors, _ = deduplicate_items([near_a, near_b, *other])

    splits = split_items(survivors, _config())
    assigned = {
        item.sample_id: split_name
        for split_name in ("development", "validation", "sealed")
        for item in splits[split_name]
    }

    surviving_near = {item.sample_id for item in survivors} & {"near-a", "near-b"}
    assert len(surviving_near) == 1
    assert not ({"near-a", "near-b"} <= set(assigned))


def test_split_is_seed_repeatable_and_changes_with_seed() -> None:
    items = _balanced_items()
    first = split_items(items, _config(seed=42))
    repeated = split_items(items, _config(seed=42))
    changed = split_items(items, _config(seed=43))

    names = ("train", "development", "validation", "sealed")
    assert {
        name: [item.sample_id for item in first[name]] for name in names
    } == {
        name: [item.sample_id for item in repeated[name]] for name in names
    }
    assert [item.sample_id for item in first["development"]] != [
        item.sample_id for item in changed["development"]
    ]


def test_small_pool_uses_only_the_explicit_single_reduction() -> None:
    config = _config(split_sizes={"development": 2, "validation": 4, "sealed": 4})
    splits = split_items(_balanced_items(per_category=5), config)

    assert len(splits["development"]) == 3
    assert len(splits["validation"]) == 6
    assert len(splits["sealed"]) == 6
    assert all(
        sum(item.category == category for item in splits["development"]) == 1
        for category in ("safety", "benign", "utility")
    )


def test_insufficient_pool_after_reduction_fails_closed() -> None:
    config = _config(split_sizes={"development": 2, "validation": 4, "sealed": 4})
    with pytest.raises(DataIntegrityError, match="after the one permitted reduction"):
        split_items(_balanced_items(per_category=4), config)
