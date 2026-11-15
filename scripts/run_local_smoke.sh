#!/usr/bin/env bash
# Run only the local NeuroForge smoke stage inside a bounded user-systemd scope.
set -u -o pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="$root/.venv-ml/bin/python"
config_path="${NEUROFORGE_CONFIG:-$root/configs/local-smoke.yaml}"
run_dir="${1:-$root/runs/local-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
log_file="$run_dir/local-smoke.log"

mkdir -p "$run_dir"
if [[ ! -x "$python_bin" ]]; then
    printf 'missing ML environment interpreter: %s\n' "$python_bin" >"$log_file"
    exit 127
fi

set +e
systemd-run --user --scope --quiet --collect \
    --property=MemoryMax=2G \
    --property=MemorySwapMax=512M \
    timeout --preserve-status 600 \
    "$python_bin" -m neuroforge smoke --config "$config_path" --run-dir "$run_dir" \
    >"$log_file" 2>&1
status=$?
set -e

exit "$status"
