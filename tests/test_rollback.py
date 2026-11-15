from __future__ import annotations

import numpy as np
import pytest

from neuroforge.artifacts import (
    ArtifactIntegrityError,
    apply_named_patch,
    immutable_snapshot,
    restore_snapshot,
)


def test_numpy_named_patch_and_rollback_are_exact() -> None:
    # NumPy toy-tensor proof only; it is explicitly not Tiny Aya model integration evidence.
    original = {
        "q_proj.weight": np.array([[1.0, 2.0]], dtype=np.float32),
        "v_proj.weight": np.array([[3.0, 4.0]], dtype=np.float32),
    }
    snapshot = immutable_snapshot(original)
    replacement = {"q_proj.weight": np.array([[5.0, 6.0]], dtype=np.float32)}

    working = apply_named_patch(original, replacement)
    working["q_proj.weight"][0, 0] = 9.0
    restored = restore_snapshot(working, snapshot)

    assert np.array_equal(original["q_proj.weight"], np.array([[1.0, 2.0]], dtype=np.float32))
    assert np.array_equal(restored["q_proj.weight"], snapshot["q_proj.weight"])
    assert np.array_equal(restored["v_proj.weight"], snapshot["v_proj.weight"])
    assert restored["v_proj.weight"] is not snapshot["v_proj.weight"]
    assert working["v_proj.weight"] is original["v_proj.weight"]


def test_snapshot_is_immune_to_working_state_mutation() -> None:
    state = {"q_proj.weight": np.array([1.0, 2.0], dtype=np.float32)}
    snapshot = immutable_snapshot(state)
    state["q_proj.weight"][0] = 42.0

    assert np.array_equal(snapshot["q_proj.weight"], np.array([1.0, 2.0], dtype=np.float32))


@pytest.mark.parametrize(
    ("replacement", "reason"),
    [
        ({"missing.weight": np.ones((1,), dtype=np.float32)}, "unknown"),
        ({"q_proj.weight": np.ones((2,), dtype=np.float32)}, "shape mismatch"),
        ({"q_proj.weight": np.ones((1,), dtype=np.float64)}, "dtype mismatch"),
        ({"q_proj.weight": np.array([np.nan], dtype=np.float32)}, "nonfinite"),
    ],
)
def test_invalid_replacement_fails_before_mutation(
    replacement: dict[str, np.ndarray], reason: str
) -> None:
    state = {"q_proj.weight": np.array([1.0], dtype=np.float32)}
    original = state["q_proj.weight"].copy()

    with pytest.raises(ArtifactIntegrityError, match=reason):
        apply_named_patch(state, replacement)

    assert np.array_equal(state["q_proj.weight"], original)
