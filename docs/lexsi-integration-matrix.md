# Lexsi integration matrix

**Scope:** exact intended integrations and their evidence boundary. “Installed” and “imported” are deliberately distinct from “executed on Tiny Aya.” No entry below implies a model load, generation, training step, repair, or measured metric unless the status cell says so.

## Status summary

| Status | Meaning | Current evidence |
|---|---|---|
| Inspected | Pinned source/readme/API was read. | All six Lexsi repositories below were assessed in the approved source audit; primary anchors for the four active toolkits were reread. |
| Installed | Resolved in an isolated environment. | CuratorKIT `0.1.21` and AuditKIT `1.1.12` are installed in `.venv-data` at the listed pins. |
| Imported | Named local Python symbols/signature were loaded/inspected. | CuratorKIT and AuditKIT only, as recorded below. |
| Executed on Tiny Aya | A real invocation on the pinned core model with retained output exists. | **None.** |

## Matrix

| Repository / pin | Planned install command (not evidence of execution) | Actual API/reference selected | Intended role | Import evidence | Tiny Aya / hardware status | Limitation and decision |
|---|---|---|---|---|---|---|
| [SafeTune](https://github.com/Lexsi-Labs/SafeTune) `f86cf77e0de50756f3885673b2e472705177722e` | `uv pip install "safetune @ git+https://github.com/Lexsi-Labs/SafeTune.git@f86cf77e0de50756f3885673b2e472705177722e"` | `safetune.interventions.recover.safety_vector_restore.apply_safety_vector_restore(model, aligned, alpha=1.0, rank=8, target_modules=None, strict=False)`; `apply_ctheta_from_state_dicts(target, positive_sd, negative_sd, circuit_info, strength=1.0, layer_subset=None, target_modules=None)`; `RefusalDirectionTrainer` for M3. | M1/M4 vector restore; M2/random C-ΔΘ matrix restoration; M3 direction steering. | Inspected pinned `safety_vector_restore.py` and `ctheta.py`; **not installed/imported** in this record. | No Tiny Aya run; no GPU use. | Vector restore mutates in place and falls back to full delta on SVD error; realized edit/rank must be verified. C-ΔΘ matches whole `.weight` matrices, errors on empty masks, and is not neuron-level. LSAL v1.2 source-available, not OSI open source. |
| [CircuitKIT](https://github.com/Lexsi-Labs/CircuitKIT) `601a5d392bab32aa600de5db12abea1cd6b2b6ec` | `uv pip install "circuitkit @ git+https://github.com/Lexsi-Labs/CircuitKIT.git@601a5d392bab32aa600de5db12abea1cd6b2b6ec"` | Import CircuitKIT before TransformerLens; `ck.discover(model, task="neuroforge_safety", algorithm="eap-ig", level="node", n_examples=8, batch_size=1, sparsity=0.1, scope="heads", ig_steps=3, signed=True, output_path=...)`; `SafetyPromptAdapter.adapt`; `random_selector`. | D3 circuit discovery; four-layer ranking; random controls. | Inspected pinned README; **not installed/imported** in this record. | Readme describes Tiny Aya as `cohere2` experimental support, but NeuroForge has not verified it. No model/GPU run. | Must pass Tiny Aya HF/TransformerLens parity (KL `<1e-4`) before circuit use. Keep the library’s Tiny Aya claim distinct from project verification. LSAL v1.2 source-available, not OSI open source. |
| [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT) `86bbacc89b0516cfb30c78e2d8073ebdf0dad0bb` / `0.1.21` | `uv pip install "curatorkit @ git+https://github.com/Lexsi-Labs/CuratorKIT.git@86bbacc89b0516cfb30c78e2d8073ebdf0dad0bb"` | `Curator(CuratorConfig(...)).run()`; `ExactDeduplicator`; `MinHashDeduplicator(threshold=.85, ngram=3, num_perm=128, seed=42, ...)`. | Prompt-only exact + MinHash pre-split leakage barrier, provenance, split materialization. | `.venv-data`: imported `curatorkit 0.1.21`; `Curator`, `CuratorConfig`, `ExactDeduplicator`, and `MinHashDeduplicator` succeeded. `inspect.signature` confirmed `Curator(config)` and the stated MinHash parameters. | Data-only; no Tiny Aya requirement. No Curator run on a fixture or study data yet. | CuratorKIT’s generic pipeline is wrapped only to preserve prompt-only cross-pool dedup, source mappings, conflict quarantine, and frozen manifests. No custom deduplicator. |
| [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) `9d4e21bac240e8108a22c2c137d179d6070868d5` / `1.1.12` | `uv pip install "auditkit @ git+https://github.com/Lexsi-Labs/AuditKIT.git@9d4e21bac240e8108a22c2c137d179d6070868d5"` | `Sample(input=..., target=..., actual_output=...)`; `auditkit.evaluate(samples, model="precomputed", scorers=[...])`; `GuardJudge(profile="granite_guardian", judge_model="hf:ibm-granite/granite-guardian-3.1-2b", assess="response", policy="harm", judge_model_args=...)`; `refusal_scorer`; `Equals`. | Independent safety/over-refusal/utility scoring, complete-coverage decisions. | `.venv-data`: imported `auditkit 1.1.12`, `GuardJudge`, and `Equals`. `inspect.signature` confirmed `GuardJudge(profile, *, judge_model, assess, policy, judge_model_args, ...)`. | No Tiny Aya, guardian, or AuditKIT evaluation model has been loaded; no scorer invocation or metric exists. | Reuse AuditKIT’s judge/template/parser and metric implementations. NeuroForge owns only gate thresholding/isolation/provenance; it does not create a custom safety judge. LSAL-1.2 source-available, non-commercial. |
| [AlignTune](https://github.com/Lexsi-Labs/aligntune) `b6e924e00d2b299f0eebb87429bd82ef2eccc896` | Not installed for this study. | Inspected only; no selected API. | Not used. | Inspected in approved source audit; **not installed/imported/executed**. | No Tiny Aya claim. | Direct TRL/PEFT is intentionally chosen because it implements the frozen recipe with a smaller dependency surface. Do not report AlignTune as used. |
| [AgentTune](https://github.com/Lexsi-Labs/AgentTune) `2aecde44d5b3f8e9752af42dc9900b7646647583` | Not installed for this study. | Inspected only; no selected API. | Not used. | Inspected in approved source audit; **not installed/imported/executed**. | No Tiny Aya claim. | No agent post-training is required for core study. Optional WSP is weightspace-based and separate, not an AgentTune integration. |

## Standard ecosystem and direct training path

| Component | Planned use | Status/constraint |
|---|---|---|
| TRL / PEFT | `SFTTrainer` with LoRA for the six frozen fault recipes; merge via `merge_and_unload(safe_merge=True)`. | Chosen instead of AlignTune. No import/training claim in this matrix. |
| Transformers / Accelerate / bitsandbytes | Pinned Tiny Aya loading and local NF4 smoke or funded BF16 cloud inference. | No model load. The workstation has roughly 524 MiB available RAM with swap nearly full at the recorded update, so local fine-tuning/CircuitKIT is not assumed feasible. |
| PyTorch / safetensors | Tensor arithmetic and safe patch artifacts; no pickle. | No model/tensor work has run. |
| TransformerLens 3.8 | `run_with_cache` and `generic_activation_patch` with `layer_pos_patch_setter` for cross-checkpoint diagnostic D2. | No import/parity test. Use only if the frozen parity gate passes. |
| SciPy / scikit-learn | Proxy correlation, bootstrap, fold-local `SimpleImputer` and depth-2 decision tree router. | No fit/result exists. |
| Pydantic v2 / Typer | Contracts and CLI. | Package skeleton is outside this matrix; no CLI execution claim. |

## Pinned model and data references

- Core model: `CohereLabs/tiny-aya-global` revision `af89d219b53ed9b13b8a4645f9c8028973510324`, CC-BY-NC-4.0, authenticated access confirmed previously only by config dry-run. It has **not** been loaded here.
- Safety/over-refusal prompts: `paul-rottger/xstest` commit `d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d`, CC-BY-4.0.
- Utility/training data: `openai/gsm8k` `main` revision `740312add88f781978c0658806c59bc2815b9866`, MIT.
- Safety guardian: `ibm-granite/granite-guardian-3.1-2b` revision `81145486e85c6c82c01e759c0356d9d6da4d21a5`, planned only.

## Required evidence before any “integrated” claim

An integration statement must attach: exact environment lock/version, import result, real API call, pinned model/data revision, hardware/peak memory, output/log hashes, and known limitation. “Inspected,” “imported,” and “executed on Tiny Aya” are never interchangeable. As of this record, no weights, metrics, model outputs, Hub uploads, cloud resources, or cloud spend were created.
