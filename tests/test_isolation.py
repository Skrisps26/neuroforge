import json
import os
import subprocess
from pathlib import Path

import pytest

from neuroforge.evaluation import build_evaluator_command, build_isolation_probe_command


def _request(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    request = inputs / "request.json"
    request.write_text(json.dumps({"config_hashes": {"study": "fixture"}, "items_path": "items.jsonl", "responses_path": "responses.json"}), encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    guardian = tmp_path / "guardian"
    base = tmp_path / "base"
    fault_adapter = tmp_path / "fault-adapter"
    candidate = tmp_path / "candidate"
    for model_root in (guardian, base, fault_adapter, candidate):
        model_root.mkdir()
    return inputs, request, output, guardian, base, fault_adapter, candidate


def test_evaluator_command_uses_fixed_unprivileged_boundary(tmp_path: Path) -> None:
    inputs, request, output, guardian, base, fault_adapter, candidate = _request(tmp_path)
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    argv = build_evaluator_command(
        image="python:3.12-slim",
        evaluator_source=Path("src"),
        inputs_root=inputs,
        input_json=request,
        output_dir=output,
        sealed_inputs_root=sealed,
        guardian_model=guardian,
        base_model=base,
        fault_adapter=fault_adapter,
        candidate_root=candidate,
        use_gpu=True,
    )
    assert argv[:7] == ["podman", "run", "--rm", "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges"]
    assert "--env-host" not in argv
    assert "--privileged" not in argv
    assert argv[argv.index("--device") + 1] == "nvidia.com/gpu=all"
    assert "/sealed" in " ".join(argv)
    assert "/app" in " ".join(argv)
    assert "/models/guardian" in " ".join(argv)
    assert "/models/base" in " ".join(argv)
    assert "/models/fault-adapter" in " ".join(argv)
    assert "/candidate" in " ".join(argv)
    assert argv[-6:] == [
        "-m",
        "neuroforge.evaluation",
        "--input",
        "/inputs/request.json",
        "--output",
        "/output/metrics.json",
    ]


def test_evaluator_command_rejects_option_and_path_injection(tmp_path: Path) -> None:
    inputs, request, output, guardian, base, fault_adapter, candidate = _request(tmp_path)
    with pytest.raises(ValueError, match="image"):
        build_evaluator_command(
            image="python:3.12-slim --privileged",
            evaluator_source=Path("src"),
            inputs_root=inputs,
            input_json=request,
            output_dir=output,
            guardian_model=guardian,
            base_model=base,
            fault_adapter=fault_adapter,
            candidate_root=candidate,
        )
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        build_evaluator_command(
            image="python:3.12-slim",
            evaluator_source=Path("src"),
            inputs_root=inputs,
            input_json=outside,
            output_dir=output,
            guardian_model=guardian,
            base_model=base,
            fault_adapter=fault_adapter,
            candidate_root=candidate,
        )


def test_probe_command_excludes_sealed_data_and_gate_configuration(tmp_path: Path) -> None:
    readonly_probe = tmp_path / "readonly-probe"
    readonly_probe.write_text("immutable", encoding="utf-8")
    argv = build_isolation_probe_command(
        image="python:3.12-slim",
        evaluator_source=Path("src"),
        readonly_probe_file=readonly_probe,
    )
    rendered = " ".join(argv)
    assert "--network=none" in argv
    assert "/sealed" not in rendered
    assert "/probe/readonly-gate" in rendered
    assert argv[-2] == "-c"
    assert "neuroforge.evaluation" not in argv


@pytest.mark.skipif(
    os.environ.get("NEUROFORGE_RUN_PODMAN_TESTS") != "1",
    reason="opt-in boundary probe requires a locally available Podman image",
)
def test_podman_isolation_probe_reports_both_denials(tmp_path: Path) -> None:
    readonly_probe = tmp_path / "readonly-probe"
    readonly_probe.write_text("immutable", encoding="utf-8")
    argv = build_isolation_probe_command(
        image=os.environ.get("NEUROFORGE_PODMAN_IMAGE", "python:3.12-slim"),
        evaluator_source=Path("src"),
        readonly_probe_file=readonly_probe,
    )
    completed = subprocess.run(argv, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout)["passed"] is True
