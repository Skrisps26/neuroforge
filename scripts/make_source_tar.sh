#!/usr/bin/env bash
# WHY CUSTOM: uv/pip build envs but no library defines this project's exact
# reproducible cloud bundle (code + configs + frozen requirements, never
# records/secrets/venvs/weights). Single definition prevents job-to-job drift.
set -euo pipefail
out="${1:?usage: make_source_tar.sh <output.tar>}"
model_stage="runs/.bundle-model"
rm -rf "$model_stage"
mkdir -p "$model_stage/models/tiny-aya-global"
cp models/tiny-aya-global/config.json models/tiny-aya-global/generation_config.json models/tiny-aya-global/model.safetensors.index.json models/tiny-aya-global/special_tokens_map.json models/tiny-aya-global/tokenizer.json models/tiny-aya-global/tokenizer_config.json "$model_stage/models/tiny-aya-global/"
# Frozen input splits ship explicitly (never records, adapters, or run dirs).
stage="runs/.bundle-stage"
rm -rf "$stage"
mkdir -p "$stage/runs/core-curation/data"
cp runs/core-curation/data/manifest.json runs/core-curation/data/train.jsonl runs/core-curation/data/development.jsonl runs/core-curation/data/validation.jsonl runs/core-curation/data/sealed.jsonl "$stage/runs/core-curation/data/"
tar --exclude='.venv*' --exclude='runs' --exclude='.git' --exclude='models' \
  --exclude='__pycache__' --exclude='.bundle-stage' -cf "$out" src scripts configs reports pyproject.toml README.md
tar -rf "$out" -C "$stage" runs
tar -rf "$out" -C "$model_stage" models
rm -rf "$stage" "$model_stage"
sha256sum "$out" | tee "$out.sha256"
