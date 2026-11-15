#!/usr/bin/env bash
# WHY CUSTOM: uv, pip and huggingface_hub implement environments and downloads;
# this script only composes the pinned NeuroForge deployment steps.
set -euo pipefail
cd /workspace/neuroforge
git config --global --add safe.directory /workspace/neuroforge
export UV_HTTP_TIMEOUT=180 UV_CACHE_DIR=/opt/neuroforge-uv-cache
export UV_PYTHON_INSTALL_DIR=/opt/neuroforge-python
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
python3 -m pip install --disable-pip-version-check --no-deps \
  --target /opt/neuroforge-bootstrap 'uv==0.11.31'
export PATH="/opt/neuroforge-bootstrap/bin:$PATH"
export PYTHONPATH="/opt/neuroforge-bootstrap${PYTHONPATH:+:$PYTHONPATH}"
uv python install 3.12.13
uv venv /opt/neuroforge-venv --python 3.12.13
uv pip install --python /opt/neuroforge-venv/bin/python \
  --default-index https://pypi.org/simple \
  --index https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match -r reports/requirements-ml.freeze.txt
uv pip check --python /opt/neuroforge-venv/bin/python
/opt/neuroforge-venv/bin/python -u - <<'PY'
import json, torch
from huggingface_hub import snapshot_download
assert torch.cuda.is_available(), 'CUDA unavailable on provisioned GPU'
print(json.dumps({'torch':torch.__version__, 'cuda':torch.version.cuda,
 'gpu':torch.cuda.get_device_name(0),
 'vram_bytes':torch.cuda.get_device_properties(0).total_memory}), flush=True)
snapshot_download('ibm-granite/granite-guardian-3.1-2b',
 revision='81145486e85c6c82c01e759c0356d9d6da4d21a5',
 local_dir='models/granite-guardian', token=False, max_workers=2,
 allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
print('PUBLIC_GUARDIAN_STAGED', flush=True)
PY
/opt/neuroforge-venv/bin/python -m neuroforge doctor --run-dir runs/cloud-doctor
