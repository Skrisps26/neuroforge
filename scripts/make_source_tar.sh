#!/usr/bin/env bash
# WHY CUSTOM: uv/pip build envs but no library defines this project's exact
# reproducible cloud bundle (code + configs + frozen requirements, never
# runs/secrets/venvs/weights). Single definition prevents job-to-job drift.
set -euo pipefail
out="${1:?usage: make_source_tar.sh <output.tar>}"
tar --exclude='.venv*' --exclude='runs' --exclude='.git' --exclude='models' \
  --exclude='__pycache__' -cf "$out" src scripts configs reports pyproject.toml README.md
sha256sum "$out" | tee "$out.sha256"
