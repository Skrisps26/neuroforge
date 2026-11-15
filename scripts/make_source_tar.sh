#!/usr/bin/env bash
# WHY CUSTOM: uv/pip build envs but no library defines this project's exact
# reproducible cloud bundle (code + configs + frozen requirements, never
# records/secrets/venvs/weights). Single definition prevents job-to-job drift.
set -euo pipefail
out="${1:?usage: make_source_tar.sh <output.tar>}"
# Frozen input splits ship explicitly (never records, adapters, or run dirs).
stage="runs/.bundle-stage"
rm -rf "$stage"
mkdir -p "$stage/runs/core-curation/data"
cp runs/core-curation/data/manifest.json runs/core-curation/data/train.jsonl runs/core-curation/data/development.jsonl runs/core-curation/data/validation.jsonl runs/core-curation/data/sealed.jsonl "$stage/runs/core-curation/data/"
tar --exclude='.venv*' --exclude='runs' --exclude='.git' --exclude='models' \
  --exclude='__pycache__' --exclude='.bundle-stage' -cf "$out" src scripts configs reports pyproject.toml README.md
tar -rf "$out" -C "$stage" runs
rm -rf "$stage"
sha256sum "$out" | tee "$out.sha256"
