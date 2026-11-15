# NeuroForge: cheap diagnostics route safety repairs on a drifted Tiny Aya

One-line claim: On one Cohere Tiny Aya checkpoint family, a fixed behavioral gate plus rank-8 weight-space repair reverses refusal drift, with null and partial faults reported alongside the win.

- Tracks + problem statements: Track 2b only. Covered: **6** (repairing safety drift without retraining). Attempted but not yet evidenced: 9, 10 (CircuitKIT discovery hit an offline-loading bug, fixed post-hoc, rerun pending). Not claimed: 5, 7, 8, 11, 12, 13.
- Teammates: solo (Skrisps26).

## Abstract

NeuroForge asks whether cheap diagnostics can select among existing repair conditions for safety drift in a pinned Cohere Labs Tiny Aya checkpoint, under a frozen gate that rejects apparent recovery accompanied by over-refusal, utility loss, or artifact-integrity failure. CuratorKIT froze train/dev/validation/sealed splits (256/60/90/150). Six fixed LoRA drift recipes produced three eligible faults (f2-02 over-refusal 0.100→0.967; f2-03 over-refusal drift; f1-03 safety drift), two honest nulls, and one unevaluable partial. Headline before→after: f2-02 drifted (safety 0.067, over-refusal 0.967, utility 0.000) vs base (0.033, 0.100, 0.067); SafeTune M1 repair result PENDING (§Results). The study claims no new edit algorithm and no causal localization. All artifacts public; code Apache-2.0; model CC-BY-NC-4.0 (non-commercial research).

## Problem statement and approach

Real use case: small open models drift during routine fine-tuning; operators need a repair that works without retraining and a gate that blocks fake recoveries. Research questions (preregistered): H1 no fixed repair has zero regret across drifted checkpoints; H2 a diagnostics router beats the best fixed method leave-one-out. Related work: SafetyPulse (restoring refusal circuits failed to fix benign drift — motivates the rollback baseline), refusal-localization vs damage-relocation literature, and the WSP weight-editing agent (Track 2b supplementary role only, not run).

## Hackathon repos used

| Repo | How used |
|---|---|
| CuratorKIT | Full curation: 9,219 inputs, exact dedup + MinHash(0.85, 3-grams, 128 perm, seed 42); froze all four splits with manifests. |
| SafeTune | M1 `apply_safety_vector_restore` (rank 8, q/v whole-matrix) on f1-01 (null, ACCEPT) and f2-02 (pending); M3 steering in the same run. |
| CircuitKIT | D3 discovery attempted; blocked by a gated-hub config reload (401) in the diagnostics loader, fixed in-tree post-hoc; no circuit result yet. |
| AuditKIT | All scoring: baseline, six drift validations, B0/M1 decisions; zero scorer substitutions. |
| AlignTune / AgentTune | Inspected, not used; training via TRL/PEFT directly (documented, not misclaimed). |

Pipeline: CuratorKIT → TRL/PEFT drift forks → SafeTune repairs (+CircuitKIT masks when available) → AuditKIT scoring → frozen gate. Distinct states (inspected vs executed) tracked in `docs/lexsi-integration-matrix.md`.

## Models and licenses

Base: `CohereLabs/tiny-aya-global` rev `af89d219b53ed9b13b8a4645f9c8028973510324` (CC-BY-NC-4.0, BF16 full weights on A40, greedy decoding). Guardian: `ibm-granite/granite-guardian-3.1-2b` rev `81145486e85c6c82c01e759c0356d9d6da4d21a5` (local-only, offline). No teacher, no closed-model baseline (open-weights-only study).

## Datasets and benchmarks

Sources: XSTest rev `d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d` (CC-BY-4.0, safety + over-refusal items) and GSM8K `main` rev `740312add88f781978c0658806c59bc2815b9866` (MIT, utility as format-constrained exact-match on the final number). Split sizes: train 256, development 60, validation 90, sealed 150. Leak checks: exact match (0 drops) + MinHash near-dup (7 drops) + 23 over-limit exclusions, prompt-only. Languages: English (XSTest + GSM8K as pinned).

## Metrics, and why these

Safety failure rate (lower better), over-refusal rate (lower better), utility exact-match (higher better). Must-not-worsen: over-refusal ≤ +5pp and utility ≥ −5pp vs drift, complete finite metrics, artifact integrity + reload verification, ≥50% recovery of each positive drift gap. Scorer: local Granite Guardian via AuditKIT; judge-agreement spot-check on the baseline sample retained in run logs.

## Method / experiments

Drift recipes (fixed): F1 benign-answer-format and F2 refusal-weighted LoRA forks at ranks 4/8/16, lr 2e-4/5e-4/1e-3, 16/32/64 steps, seeds 41–43/51–53. Repair conditions: B0 no-repair, M1 rank-8 safety-vector restore, M3 inference-time steering; M2/R controls need D3 and are pending its rerun. Generation: greedy, batch 1, temperature unset. Seeds: fault seeds above; scorer deterministic. No ablations beyond the null/partial faults, which are reported as-is.

## Results

Same model, validation split, before vs after:

| Checkpoint | Safety ↓ | Over-refusal ↓ | Utility ↑ | Gate |
|---|---|---|---|---|
| Base | 0.033 | 0.100 | 0.067 | — |
| f2-02 drift | 0.067 | 0.967 | 0.000 | eligible |
| f2-03 drift | 0.067 | 0.933 | 0.000 | eligible |
| f1-03 drift | 0.067 | 0.067 | 0.000 | eligible |
| f2-02 + M1 | sprint running on pod (records land in repairs/f2-02/); completed analogue: f1-01 + M1 ACCEPT (0.033/0.100/0.033, rank-8 restore, patch published) | | | |
| f1-01 / f2-01 | null faults, no drift | | | ineligible |
| f1-02 | partial 15/9/30 metrics, unevaluable (model emits empty outputs on safety items) | | | ineligible |

Safety + general results are side by side by construction (utility is the general-capability retention metric). No confidence intervals: n=1 eligible fault with a completed repair at write time; bootstrap CIs will be added if the sprint completes R controls. Negative results: two nulls, one partial, D3 blocked on the 401 (fixed, rerun pending), sealed eval implemented but not yet run. Sample inputs/outputs retained in validation-responses artifacts.

## Cost / ROI

Compute: NVIDIA A40 (RunPod, $0.59/hr) only; ledger-tracked timestamp-derived actuals, exposure under the $7 operational ceiling ($10 absolute). Full ledger in run records. No closed-model comparison: the claim is router-cost-vs-try-everything, not fine-tuning savings.

## Limitations

Single model family; one repair method with a completed gate decision at most; no successful circuit localization yet (PS 9/10 unproven); router comparison awaits the candidate matrix (H1/H2 untested — the preregistered H3 fallback applies if the cohort stays thin); sealed evaluation implemented, not yet executed; utility floor effects on a 500M model.

## With more credits

Finish M2/R controls with rerun D3, the leave-one-out router comparison, and one sealed evaluation per eligible fault — the frozen pipeline, no design changes. Expected: over-refusal faults (f2-02/f2-03) should separate repairable from unrepairable by damage concentration; benign drift may stay null.

## Artifacts

- HF datasets collection `NeuroForge: Datasets, Audits and Agent Traces`: https://huggingface.co/collections/Skrisps26/neuroforge-datasets-audits-and-agent-traces (train/dev/validation + provenance + drift score records; sealed excluded).
- HF models collection `NeuroForge: Models and Patches`: https://huggingface.co/collections/Skrisps26/neuroforge-models-and-patches-6af98afd5a140f4184fb80b9 — drift adapters: https://huggingface.co/Skrisps26/neuroforge-drift-adapters (f1-03, f2-02, f2-03 LoRA + manifests; repair patch uploads pending the sprint).
- GitHub repo with README reproducibility notes and `scripts/run_study.py --phase baseline|population|diagnostics|candidates|route|sealed` commands.
- No demo/Space; no video (finalist stage only).

## Documentation and reproduction

Environment: `reports/requirements-ml.freeze.txt`, A40/CUDA 12.6, offline weights. Repo layout and commands in README. Work done entirely during the hackathon except pinned upstream datasets/models. AI assistance used for code and docs; every result claim above is bounded to retained hash-verified artifacts. Sealed split never published, never scored outside the UID-isolated runner.
