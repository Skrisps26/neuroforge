# NeuroForge submission write-up

**Status: in progress, not submitted.** Results below are observed executions only; pending items name the exact command that will fill them. H1–H6 remain untested (no router comparison exists).

## Abstract

NeuroForge is a preregistered repair-selection study for safety drift in one pinned Cohere Labs Tiny Aya checkpoint. Its question is whether checkpoint diagnostics can select among existing repair conditions while an independent behavioral gate rejects apparent safety recovery accompanied by over-refusal, utility loss, incomplete coverage, or artifact-integrity failure. The frozen design uses prompt-only cross-source deduplication, fixed LoRA drift recipes, SafeTune repair conditions, CircuitKIT-derived localization only after parity gates, AuditKIT scoring, matched random-location controls, and leave-one-checkpoint-out routing regret. The study does not claim a new edit algorithm or that an edit proves causal safety localization.

Observed so far: CuratorKIT froze train/development/validation/sealed splits of 256/60/90/150 records. A cloud A40 BF16 greedy baseline scored development (20/20/20: safety 0.0, over-refusal 0.1, utility 0.05) and validation (30/30/30: safety 0.0333, over-refusal 0.1, utility 0.0667) with zero evaluation errors. The first benign drift checkpoint (f1-01: LoRA rank 4, 16 steps, adapter reload value-equal) shows no measured safety drift (drifted safety 0.0333 equals base 0.0333), so it is an honestly reported null fault: B0 ACCEPT, ineligible for routing. M1 repair is in progress after fixing a CPU-placement stall. No circuit measurement, router comparison, sealed score, or WSP session exists. A nonsealed development-data collection is published; the models collection awaits a real accepted artifact. (Abstract: 198 words.)

## Track fit and toolkit use

The supported Track 2b problem statements are **6** (repairing safety drift without retraining), **9** (locating where safety lives), and **10** (circuit-level interventions). Problem 8 (inference-time steering) becomes supported only if an M3 condition completes; problem 5 is not claimed.

| Repository | Status in NeuroForge |
|---|---|
| CuratorKIT | Used for the completed full curation and frozen manifests. |
| SafeTune | M1 invoked on f1-01 (first attempt recorded BrokenPipeError from a dead session pipe; retry running with GPU-placed weights); M2/M3 pending diagnostics. |
| CircuitKIT | Planned diagnostic/ranking integration; no parity, discovery, or circuit result yet. |
| AuditKIT | Scoring real model outputs: baseline dev/validation plus f1-01 drifted/B0, zero evaluation errors. |
| AlignTune / AgentTune | Inspected, not used. Direct TRL/PEFT trained f1-01 (16 steps, loss trajectory retained). |

The protocol chains CuratorKIT → TRL/PEFT → SafeTune/CircuitKIT → AuditKIT. Source inspection, package import, implemented adapters, and model execution remain distinct states, tracked in `docs/lexsi-integration-matrix.md`.

## Method summary

The model is `CohereLabs/tiny-aya-global` at revision `af89d219b53ed9b13b8a4645f9c8028973510324`; the guardian is `ibm-granite/granite-guardian-3.1-2b` at `81145486e85c6c82c01e759c0356d9d6da4d21a5` (local-only, offline). XSTest is pinned to `d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d` (CC-BY-4.0) and GSM8K `main` to `740312add88f781978c0658806c59bc2815b9866` (MIT). The model is CC-BY-NC-4.0; this is non-commercial research.

Before splits, curation used exact deduplication and MinHash at threshold 0.85, character 3-grams, 128 permutations, seed 42, comparing prompts with a constant output. The curation manifest SHA-256 is `4e03e4ed07f0f41c0351ad2f1e3788bcffd6d98352cbae0b4d79ccf3ec247041` and its configuration hash is `53c631aad1b9db410a2055d22b0a2c155e7be3c418092f6847ab4211a799b85b`.

Six fixed fault recipes (three F1, three F2) vary rank, learning rate, steps, seed, and answer format; one checkpoint (f1-01) is observed, five are pending. Conditions are B0, M1, M2, M3, M4 (tier 2), R0/R1/R2, with W1 optional. The acceptance gate requires complete finite safety, over-refusal, and utility metrics; integrity and reload verification; at least 50% recovery of each positive observed drift gap; at most +5 percentage points over-refusal and −5 points utility versus drift. The router will use five diagnostics, fold-local median imputation, and a depth-2 tree. No gate decision beyond B0/f1-01 or router comparison has been observed.

## Results

| Item | Result |
|---|---|
| Curation | completed: 9,219 inputs; 0 exact drops; 7 near drops; 23 over-limit exclusions; 9,212 survivors |
| Frozen core split counts | train 256; development 60; validation 90; sealed 150 |
| Local NF4 smoke | feasibility only: 378 input tokens; 3 new tokens; 2,694,060,544-byte peak CUDA allocation |
| Cloud BF16 baseline | dev safety 0.0 / over-refusal 0.1 / utility 0.05; validation safety 0.0333 / over-refusal 0.1 / utility 0.0667; complete, zero errors |
| f1-01 drift + B0 | adapter reload value-equal; null fault (no safety increase); B0 ACCEPT; router-ineligible |
| M1 on f1-01 | PENDING: first attempt BrokenPipeError on dead session pipe; retry with GPU placement running (`runs/cloud-proof-*/first-proof.log`) |
| Remaining faults/repairs/diagnostics/router/sealed/WSP | not completed; fill with `run_study.py --phase population|diagnostics|candidates|route` and sealed `audit` |

Utility is the preregistered format-constrained exact-match score (final number only), not a few-shot GSM8K leaderboard number; its low absolute values are expected and reported as such.

## Reproducibility, limits, and release state

Use the implemented `neuroforge curate`, `smoke`, `baseline`, `make-faults`, `repair`, `audit`, and `route` commands plus `scripts/run_study.py --phase baseline|first-proof|population|diagnostics|candidates|route`. `audit` refuses sealed data; the sealed evaluator is an independent boundary with UID/file-permission isolation. Two early cloud setups failed (NFS root-squash archive ownership; a watchdog timer that fired late) and their resources were deleted with nothing experiment-relevant lost; the third run produced the baseline above. Current ledger exposure is $4.83 (five $0.70 unsettled job reservations plus $0.63 historical storage) against $7 operational / $10 absolute ceilings; settled billing lags the ledger and reservations are retained until reconciled.

Published: Hugging Face dataset collection “NeuroForge: Datasets, Audits and Agent Traces” with repository `Skrisps26/neuroforge-development-data` (commit `b5bb6b68ada9e0bd677993dab707aa47372247e4`, train/dev/validation JSONL + provenance, hash-verified at upload; sealed and extension splits excluded). No model repository or models collection yet — those require a real accepted repair artifact. The code repository is https://github.com/Skrisps26/neuroforge (push pending a `gh auth login` refresh; all work is committed locally). Original NeuroForge code is Apache-2.0 only; upstream model, data, toolkit, and artifact terms remain in force. AI assistance was used for repository development and documentation; all result claims above are bounded to retained evidence.

## With more credits

With additional compute, the study would complete the five remaining fault trainings, the M1–M3/R0–R2 candidate matrix, diagnostics D1–D3, the leave-one-out router comparison, and one sealed evaluation — the exact frozen pipeline above, no design changes. Expected (non-result) forecast: benign LoRA drift may frequently be null (as f1-01 was), so the router thesis could remain untestable even with full execution; that null outcome would be reported as prominently as any positive.
