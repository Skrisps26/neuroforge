"""Unix-UID evaluator boundary for hosts where nested Podman is unavailable.

This backend is intentionally thin: it stages immutable inputs under one root-owned
working directory and runs the existing evaluator as a distinct unprivileged UID.
It is a process/filesystem boundary, not a network-security claim.
"""

from __future__ import annotations

import grp
import hashlib
import json
import os
import pwd
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

GATE_UID = 2100
REPAIR_UID = 2101
GATE_USER = "neuroforge-gate"
REPAIR_USER = "neuroforge-repair"
SEAL_ROOT = Path("/opt/neuroforge-sealed")
_MARKER = ".neuroforge-isolation-owned"
_JOB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class IsolationBlocked(RuntimeError):
    """Raised when the required OS authority boundary cannot be established."""


@dataclass(frozen=True)
class IsolationPaths:
    """Root-owned paths prepared for exactly one evaluator invocation."""

    root: Path
    runtime: Path
    request: Path
    output: Path
    source: Path


@dataclass(frozen=True)
class IsolationRun:
    """Observed subprocess result; network isolation is reported, never assumed."""

    returncode: int
    stdout: str
    stderr: str
    network_namespace: bool


def _require_root() -> None:
    if os.geteuid() != 0:
        raise IsolationBlocked("Unix UID evaluator boundary requires a root parent process")


def _trusted_tool(*candidates: str) -> str:
    for candidate in candidates:
        path = Path(candidate)
        try:
            details = path.stat()
        except FileNotFoundError:
            continue
        if path.is_file() and details.st_uid == 0 and not (stat.S_IMODE(details.st_mode) & 0o022):
            return str(path)
    raise IsolationBlocked(f"required trusted executable is unavailable: {candidates[0]}")


def _identity_by_uid(uid: int) -> pwd.struct_passwd | None:
    matches = [entry for entry in pwd.getpwall() if entry.pw_uid == uid]
    if len(matches) > 1:
        raise IsolationBlocked(f"UID {uid} is assigned to multiple accounts")
    return matches[0] if matches else None


def _ensure_identity(*, user: str, uid: int) -> None:
    _require_root()
    existing = pwd.getpwnam(user) if user in {entry.pw_name for entry in pwd.getpwall()} else None
    holder = _identity_by_uid(uid)
    if existing is not None:
        if existing.pw_uid != uid:
            raise IsolationBlocked(f"{user} exists with UID {existing.pw_uid}, expected {uid}")
        if holder is None or holder.pw_name != user:
            raise IsolationBlocked(f"UID {uid} is not uniquely assigned to {user}")
        return
    if holder is not None:
        raise IsolationBlocked(f"UID {uid} is already assigned to {holder.pw_name}")

    group = user
    existing_group = (
        grp.getgrnam(group) if group in {entry.gr_name for entry in grp.getgrall()} else None
    )
    if existing_group is not None and existing_group.gr_gid != uid:
        raise IsolationBlocked(f"{group} exists with GID {existing_group.gr_gid}, expected {uid}")
    if existing_group is None:
        groupadd = _trusted_tool("/usr/sbin/groupadd", "/usr/bin/groupadd")
        subprocess.run(
            [groupadd, "--gid", str(uid), group], check=True, env={"PATH": "/usr/sbin:/usr/bin"}
        )
    useradd = _trusted_tool("/usr/sbin/useradd", "/usr/bin/useradd")
    subprocess.run(
        [
            useradd,
            "--system",
            "--uid",
            str(uid),
            "--gid",
            group,
            "--no-create-home",
            "--shell",
            "/usr/sbin/nologin",
            user,
        ],
        check=True,
        env={"PATH": "/usr/sbin:/usr/bin"},
    )
    created = pwd.getpwnam(user)
    if created.pw_uid != uid or _identity_by_uid(uid) is None:
        raise IsolationBlocked(f"failed to create {user} with UID {uid}")


def ensure_isolation_identities() -> None:
    """Create the two dedicated ephemeral-pod identities, or verify exact matches."""
    _ensure_identity(user=GATE_USER, uid=GATE_UID)
    _ensure_identity(user=REPAIR_USER, uid=REPAIR_UID)


def _lstat_regular(path: Path, *, directory: bool | None = None) -> os.stat_result:
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode):
        raise ValueError(f"symlinks are forbidden in isolation inputs: {path}")
    if directory is True and not stat.S_ISDIR(details.st_mode):
        raise ValueError(f"expected directory: {path}")
    if directory is False and not stat.S_ISREG(details.st_mode):
        raise ValueError(f"expected regular file: {path}")
    if directory is None and not (stat.S_ISDIR(details.st_mode) or stat.S_ISREG(details.st_mode)):
        raise ValueError(f"unsupported filesystem object: {path}")
    return details


def _readonly_tree(source: Path | str, *, label: str) -> Path:
    raw = Path(source).expanduser()
    _lstat_regular(raw, directory=True)
    root = raw.resolve(strict=True)
    root_details = _lstat_regular(root, directory=True)
    nobody = pwd.getpwnam("nobody").pw_uid
    if root_details.st_uid not in {0, nobody} or stat.S_IMODE(root_details.st_mode) & 0o022:
        raise IsolationBlocked(f"{label} must be root/nobody-owned and non-writable: {root}")
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in [*directories, *files]:
            path = Path(parent) / name
            details = _lstat_regular(path)
            mode = stat.S_IMODE(details.st_mode)
            if details.st_uid not in {0, nobody} or mode & 0o022:
                raise IsolationBlocked(
                    f"{label} must be root/nobody-owned and non-writable: {path}"
                )
            required = 0o005 if stat.S_ISDIR(details.st_mode) else 0o004
            if mode & required != required:
                raise IsolationBlocked(f"{label} is not readable by the gate UID: {path}")
    if stat.S_IMODE(root_details.st_mode) & 0o005 != 0o005:
        raise IsolationBlocked(f"{label} is not readable by the gate UID: {root}")
    return root


def _immutable_tree(source: Path | str, *, label: str) -> Path:
    """Accept a frozen repair artifact without requiring the repair worker's UID."""
    raw = Path(source).expanduser()
    _lstat_regular(raw, directory=True)
    root = raw.resolve(strict=True)
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in [Path(parent), *(Path(parent) / entry for entry in [*directories, *files])]:
            details = _lstat_regular(name)
            if stat.S_IMODE(details.st_mode) & 0o222:
                raise IsolationBlocked(f"{label} must be immutable before evaluation: {name}")
    return root


def _copy_tree_bytes(source: Path, destination: Path, *, owner: int, immutable: bool) -> str:
    """Copy regular-file bytes only, then verify an exact tree digest without parsing content."""
    if destination.exists():
        raise IsolationBlocked(f"refusing to overwrite staged path: {destination}")
    destination.mkdir(mode=0o700, parents=False)
    os.chown(destination, owner, owner)
    for parent, directories, files in os.walk(source, followlinks=False):
        relative_parent = Path(parent).relative_to(source)
        target_parent = destination / relative_parent
        for name in directories:
            source_path = Path(parent) / name
            _lstat_regular(source_path, directory=True)
            target = target_parent / name
            target.mkdir(mode=0o700)
            os.chown(target, owner, owner)
        for name in files:
            source_path = Path(parent) / name
            _lstat_regular(source_path, directory=False)
            target = target_parent / name
            with source_path.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            os.chown(target, owner, owner)
            os.chmod(target, 0o444 if immutable else 0o400)
    for parent, directories, _ in os.walk(destination, topdown=False):
        for name in directories:
            os.chmod(Path(parent) / name, 0o555 if immutable else 0o700)
    os.chmod(destination, 0o555 if immutable else 0o700)
    copied = tree_sha256(destination)
    if copied != tree_sha256(source):
        raise IsolationBlocked(f"byte verification failed while staging {source}")
    return copied


def tree_sha256(source: Path | str) -> str:
    """Hash a regular tree by relative name and byte digest without decoding its files."""
    root = Path(source).resolve(strict=True)
    _lstat_regular(root, directory=True)
    digest = hashlib.sha256()
    found = False
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in sorted([*directories, *files]):
            path = Path(parent) / name
            details = _lstat_regular(path)
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(b"D\0" if stat.S_ISDIR(details.st_mode) else b"F\0")
            digest.update(relative)
            digest.update(b"\0")
            if stat.S_ISREG(details.st_mode):
                found = True
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
    if not found:
        raise ValueError(f"tree contains no regular files: {root}")
    return digest.hexdigest()


def file_sha256(source: Path | str) -> str:
    path = Path(source).resolve(strict=True)
    _lstat_regular(path, directory=False)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze_proof(
    proof_path: Path | str, *, candidate: Path, config: Path, router_selection: Path
) -> None:
    proof = json.loads(Path(proof_path).read_text(encoding="utf-8"))
    expected_keys = {"candidate_sha256", "config_sha256", "router_selection_sha256"}
    if not isinstance(proof, dict) or set(proof) != expected_keys:
        raise ValueError(
            "freeze proof must contain exactly candidate/config/router selection hashes"
        )
    if any(
        not isinstance(proof[key], str) or not _SHA256.fullmatch(proof[key])
        for key in expected_keys
    ):
        raise ValueError("freeze proof hashes must be lower-case SHA-256 hex")
    observed = {
        "candidate_sha256": tree_sha256(candidate),
        "config_sha256": file_sha256(config),
        "router_selection_sha256": file_sha256(router_selection),
    }
    if proof != observed:
        raise IsolationBlocked(
            "freeze proof does not match candidate, config, and router selection"
        )


def _owned_workdir(working_root: Path | str, job_name: str) -> Path:
    _require_root()
    if not _JOB_NAME.fullmatch(job_name):
        raise ValueError("job name must be a short path-safe identifier")
    raw_root = Path(working_root).expanduser()
    _lstat_regular(raw_root, directory=True)
    root = raw_root.resolve(strict=True)
    if root != SEAL_ROOT and SEAL_ROOT not in root.parents:
        raise IsolationBlocked(f"sealed working root must be beneath {SEAL_ROOT}")
    details = _lstat_regular(root, directory=True)
    if details.st_uid != 0 or stat.S_IMODE(details.st_mode) & 0o022:
        raise IsolationBlocked("sealed working root must be root-owned and non-writable")
    work = root / job_name
    try:
        work.mkdir(mode=0o711)
    except FileExistsError as error:
        raise IsolationBlocked(f"refusing to reuse evaluator work directory: {work}") from error
    os.chown(work, 0, 0)
    marker = work / _MARKER
    marker.write_text("root-owned ephemeral evaluator workspace\n", encoding="ascii")
    os.chown(marker, 0, 0)
    os.chmod(marker, 0o400)
    return work


def _root_symlink(destination: Path, source: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise IsolationBlocked(f"refusing to replace staged path: {destination}")
    destination.symlink_to(source, target_is_directory=True)
    details = destination.lstat()
    if details.st_uid != 0:
        raise IsolationBlocked(f"root did not own staged symlink: {destination}")


def prepare_evaluator(
    *,
    working_root: Path | str = SEAL_ROOT,
    job_name: str,
    evaluator_source: Path | str,
    thresholds_root: Path | str,
    guardian_model: Path | str,
    base_model: Path | str,
    fault_adapter: Path | str | None,
    candidate_root: Path | str | None,
    inputs_root: Path | str,
    request_name: str,
    sealed_inputs_root: Path | str,
    sealed_marker: str,
    freeze_proof: Path | str | None,
    config_path: Path | str,
    router_selection_path: Path | str,
    target: str = "candidate",
) -> IsolationPaths:
    """Stage only byte-verified artifacts under an owned local-disk workspace.

    Candidate evaluation copies the fault adapter and candidate as opaque byte
    trees only after the parent-supplied freeze proof verifies them. Base
    evaluation stages neither artifact and therefore cannot load a fault adapter.
    """
    if target not in {"base", "candidate"}:
        raise ValueError("target must be base or candidate")
    if target == "candidate" and (
        fault_adapter is None or candidate_root is None or freeze_proof is None
    ):
        raise ValueError("candidate evaluation requires adapter, candidate, and freeze proof")
    if target == "base" and any(
        value is not None for value in (fault_adapter, candidate_root, freeze_proof)
    ):
        raise ValueError("base evaluation must not stage a fault adapter or candidate")
    ensure_isolation_identities()
    source = _readonly_tree(evaluator_source, label="evaluator source")
    thresholds = _readonly_tree(thresholds_root, label="thresholds")
    guardian = _readonly_tree(guardian_model, label="guardian model")
    base = _readonly_tree(base_model, label="base model")
    adapter = _immutable_tree(fault_adapter, label="fault adapter") if fault_adapter else None
    candidate = _immutable_tree(candidate_root, label="candidate") if candidate_root else None
    inputs = _readonly_tree(inputs_root, label="inputs")
    sealed = _readonly_tree(sealed_inputs_root, label="sealed inputs")
    config = Path(config_path).resolve(strict=True)
    router_selection = Path(router_selection_path).resolve(strict=True)
    _lstat_regular(config, directory=False)
    _lstat_regular(router_selection, directory=False)
    if candidate is not None:
        _freeze_proof(
            freeze_proof, candidate=candidate, config=config, router_selection=router_selection
        )
    request_relative = Path(request_name)
    if (
        request_relative.is_absolute()
        or ".." in request_relative.parts
        or request_relative.suffix != ".json"
    ):
        raise ValueError("request name must be a relative JSON path")
    request_source = inputs / request_relative
    _lstat_regular(request_source, directory=False)
    try:
        staged_target = json.loads(request_source.read_text(encoding="utf-8")).get("target")
    except (AttributeError, json.JSONDecodeError) as error:
        raise ValueError("request must be a JSON object with an approved target") from error
    if staged_target != target:
        raise ValueError("request target does not match the staged evaluator target")
    marker_relative = Path(sealed_marker)
    if marker_relative.is_absolute() or ".." in marker_relative.parts:
        raise ValueError("sealed marker must be a relative path")
    _lstat_regular(sealed / marker_relative, directory=False)

    work = _owned_workdir(working_root, job_name)
    runtime = work / "runtime"
    runtime.mkdir(mode=0o711)
    os.chown(runtime, 0, 0)
    models = runtime / "models"
    models.mkdir(mode=0o711)
    os.chown(models, 0, 0)
    _root_symlink(runtime / "thresholds", thresholds)
    _root_symlink(models / "guardian", guardian)
    _root_symlink(models / "base", base)
    if adapter is not None:
        _copy_tree_bytes(adapter, models / "fault-adapter", owner=0, immutable=True)
    if candidate is not None:
        _copy_tree_bytes(candidate, runtime / "candidate", owner=0, immutable=True)
    _copy_tree_bytes(inputs, runtime / "inputs", owner=0, immutable=True)
    _copy_tree_bytes(sealed, runtime / "sealed", owner=GATE_UID, immutable=False)
    output = runtime / "output"
    output.mkdir(mode=0o700)
    os.chown(output, GATE_UID, GATE_UID)
    gate_file = runtime / "gate-probe"
    gate_file.write_text("gate-only probe\n", encoding="ascii")
    os.chown(gate_file, GATE_UID, GATE_UID)
    os.chmod(gate_file, 0o600)
    return IsolationPaths(
        root=work,
        runtime=runtime,
        request=runtime / "inputs" / request_relative,
        output=output / "metrics.json",
        source=source,
    )


def _trusted_gpu_library_path() -> str | None:
    directories = [Path("/usr/local/nvidia/lib64"), Path("/usr/local/nvidia/lib")]
    approved: list[str] = []
    for directory in directories:
        try:
            details = directory.stat()
        except FileNotFoundError:
            continue
        if (
            directory.is_dir()
            and details.st_uid == 0
            and not (stat.S_IMODE(details.st_mode) & 0o022)
        ):
            approved.append(str(directory))
    return ":".join(approved) if approved else None


def build_evaluator_command(
    paths: IsolationPaths, *, network_namespace: bool = False, use_gpu: bool = False
) -> list[str]:
    """Build a fixed, credential-free command for the dedicated gate identity."""
    setpriv = _trusted_tool("/usr/bin/setpriv", "/bin/setpriv")
    runtime = paths.runtime.resolve(strict=True)
    if runtime != paths.root / "runtime" or paths.output.exists():
        raise IsolationBlocked("invalid or already-used isolation workspace")
    environment = [
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        f"PYTHONPATH={paths.source}",
        f"NEUROFORGE_EVALUATOR_ROOT={runtime}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "HF_DATASETS_OFFLINE=1",
        "TRUST_REMOTE_CODE=false",
        "PYTHONDONTWRITEBYTECODE=1",
        "HOME=/nonexistent",
    ]
    if use_gpu:
        environment.append("NEUROFORGE_EVALUATOR_GPU=1")
    library_path = _trusted_gpu_library_path()
    if library_path is not None:
        environment.append(f"LD_LIBRARY_PATH={library_path}")
    command = ["env", "-i", *environment]
    if network_namespace:
        command.extend([_trusted_tool("/usr/bin/unshare", "/bin/unshare"), "--net", "--"])
    command.extend(
        [
            setpriv,
            f"--reuid={GATE_UID}",
            f"--regid={GATE_UID}",
            "--clear-groups",
            "--bounding-set=-all",
            "--inh-caps=-all",
            "--ambient-caps=-all",
            "--no-new-privs",
            "--",
            sys.executable,
            "-m",
            "neuroforge.evaluation",
            "--input",
            str(paths.request),
            "--output",
            str(paths.output),
        ]
    )
    return command


def _network_namespace_supported() -> bool:
    try:
        unshare = _trusted_tool("/usr/bin/unshare", "/bin/unshare")
    except IsolationBlocked:
        return False
    completed = subprocess.run(
        [unshare, "--net", "--", "true"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"},
        check=False,
    )
    return completed.returncode == 0


def run_evaluator(paths: IsolationPaths, *, use_gpu: bool = False) -> IsolationRun:
    """Run the existing evaluator only after proving root-controlled UID isolation."""
    _require_root()
    network_namespace = _network_namespace_supported()
    completed = subprocess.run(
        build_evaluator_command(paths, network_namespace=network_namespace, use_gpu=use_gpu),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"},
        check=False,
    )
    return IsolationRun(completed.returncode, completed.stdout, completed.stderr, network_namespace)


def _probe_command(uid: int, program: str) -> list[str]:
    setpriv = _trusted_tool("/usr/bin/setpriv", "/bin/setpriv")
    return [
        "env",
        "-i",
        "PATH=/usr/sbin:/usr/bin:/sbin:/bin",
        setpriv,
        f"--reuid={uid}",
        f"--regid={uid}",
        "--clear-groups",
        "--bounding-set=-all",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--no-new-privs",
        "--",
        sys.executable,
        "-c",
        program,
    ]


def probe_authority(
    paths: IsolationPaths,
    *,
    sealed_marker: str,
    source_probe_relative: str = "neuroforge/__init__.py",
) -> dict[str, bool]:
    """Prove distinct UID authority with actual failed opens, not missing-path checks."""
    _require_root()
    marker = paths.runtime / "sealed" / sealed_marker
    source = paths.source / source_probe_relative
    gate_file = paths.runtime / "gate-probe"
    _lstat_regular(marker, directory=False)
    _lstat_regular(source, directory=False)
    _lstat_regular(gate_file, directory=False)
    repair = (
        "import os,sys;"
        f"sealed={str(marker)!r};gate={str(gate_file)!r};"
        "blocked=[];"
        "\nfor path in (sealed,gate):"
        "\n try: os.open(path, os.O_RDONLY if path==sealed else os.O_WRONLY)"
        "\n except PermissionError: blocked.append(path)"
        "\n else: sys.exit(3)"
        "\nsys.exit(0 if len(blocked)==2 else 4)"
    )
    gate = (
        "import os,sys;"
        f"sealed={str(marker)!r};source={str(source)!r};"
        "\nfor path in (sealed,source): os.close(os.open(path, os.O_RDONLY))"
        "\ntry: os.open(source, os.O_WRONLY)"
        "\nexcept PermissionError: sys.exit(0)"
        "\nelse: sys.exit(5)"
    )
    repair_result = subprocess.run(
        _probe_command(REPAIR_UID, repair),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"},
    )
    gate_result = subprocess.run(
        _probe_command(GATE_UID, gate),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"},
    )
    result = {
        "repair_cannot_read_sealed_marker_or_edit_gate_file": repair_result.returncode == 0,
        "gate_can_read_marker_and_source_but_cannot_write_source": gate_result.returncode == 0,
    }
    if not all(result.values()):
        raise IsolationBlocked(f"authority probe failed: {result}")
    return result
