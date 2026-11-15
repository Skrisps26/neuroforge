# Environment inspection

Machine UTC timestamp observed during resumed inspection: `2026-11-14T18:32:40+00:00`.

## Hardware and tools

- Linux x86_64; NVIDIA RTX 2050. `nvidia-smi` reports 4096 MiB VRAM; PyTorch reports 4,007,919,616 device bytes. Driver 615.71.09.
- Host memory: approximately 7 GiB RAM and 7 GiB swap. Most recent `free -h`: 1.3 GiB available RAM, 1.1 GiB available swap. An earlier attempt had only 380 MiB available RAM and effectively full swap. Other user applications were not terminated.
- System Python 3.14.7. Project environments use Python 3.12.13 through uv.
- Verified tools: git 2.55, gh 2.94, hf 2.4, Podman 5.8.4, runpodctl 2.9.
- `.venv-data`: lightweight project dependencies, pinned CuratorKIT and AuditKIT, Transformers/tokenizer dependencies. `.venv-ml`: PyTorch 2.13.0+cu126, torchvision 0.28.0+cu126, Transformers 5.21.1, TRL 1.7.1, PEFT 0.18.0, bitsandbytes 0.49.2, accelerate 1.15, pinned SafeTune/CircuitKIT/AuditKIT.
- PyTorch, Transformers, TRL, PEFT, bitsandbytes and accelerate imports succeeded before installing the combined toolkit graph. PyTorch reported CUDA available and identified the RTX 2050. Installation alone does not establish toolkit runtime compatibility.
- An initial NVIDIA CUPTI download timed out. Retrying the exact CUDA 12.6 wheel selection with a longer HTTP timeout completed. A subsequent combined toolkit import attempt exceeded its 90-second bound without returning a traceback or success output. Runtime compatibility remains unverified; memory pressure is a possible cause, not an established diagnosis.
- No language server is configured. Python tests and runtime checks are used instead.

## Authentication and services

- GitHub and Hugging Face authenticated as `Skrisps26`. Tiny Aya gated configuration access succeeded. Credentials were not printed or committed.
- Public source repository: https://github.com/Skrisps26/neuroforge . First verified commit: `522fb22`.
- RunPod MCP and saved runpodctl authentication are available. Attempt 1 provisioned an A40 job but archive extraction failed because NFS root-squash rejected ownership restoration. Both created resources were deleted, no experiment output was lost, and no paid resource is active. A restricted rootless-Podman proof validated the `tar --no-same-owner` preparation fix for attempt 2.
- The ledger retains a $0.70 unsettled reservation from the failed/ambiguous provider request and $0.6300000152550638 historical storage exposure, for $1.3300000152550638 total exposure. That leaves $5.6699999847449362 under the $7 operational ceiling and $8.6699999847449362 under the $10 absolute ceiling. The funded balance does not authorize spending above either ceiling.
- `META_API_KEY` was absent at inspection. Optional WSP calls have not run. No paid editor API has been used.

## Staged artifacts and runtime proof limits

- Tiny Aya revision `af89d219b53ed9b13b8a4645f9c8028973510324` downloaded locally. Weight shard sizes: 4,992,396,352 and 1,706,092,176 bytes. SHA-256 respectively `0590022d0efe06f12e08e1e94302f069dff404341b426f7435c6303725635fb1` and `2ac3809931d113785058809f27775620e0b8aec6fbc7e163c9c09c08b5f90dab`.
- Pinned GSM8K train/test parquet files staged and inspected: 7473/1319 rows with `question` and `answer` fields. This is staging, not yet CuratorKIT-built evaluation data.
- Tiny Aya tokenizer ran. The harmless smoke prompt has 13 raw tokens but 378 tokens with its complete chat template. `apply_chat_template(tokenize=True)` returns a BatchEncoding in this installed version; `len(result)` is not the token count. Full-template token limits must be checked.
- The model generation configuration records EOS token 3; the tokenizer reports EOS token 6. Preserve and inspect model/template termination semantics rather than silently equating these fields.
- A local bounded NF4 smoke completed a harmless request: 378 input tokens, 3 new tokens, 2,694,060,544 bytes peak CUDA allocation, and 152 seconds process time. This is a feasibility observation only, not a baseline, training, repair, or evaluation result.
- Granite Guardian staging was interrupted after configuration files. A resumable pinned download is in progress; its presence is not yet proof of a complete model.
- Rootless Podman launched `python:3.12-slim` with no network, read-only root, dropped capabilities and no-new-privileges. Image digest: `sha256:05a994d2344eb364cc186910f4fc8786a38215f39f00d0acc25bff33a08f4177`. This verifies container launch only; forbidden-file and evaluator authority boundaries still require behavioral tests.

No baseline, drift, repair, circuit, router or final-test result is established by this environment inspection.
