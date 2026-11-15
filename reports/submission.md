# NeuroForge submission write-up

**Status: prepared, not submitted.** This is an honest pre-results write-up, not a claim of a completed Track 2b experiment.

## Abstract

NeuroForge is a preregistered repair-selection study for safety drift in one pinned Cohere Labs Tiny Aya checkpoint. Its question is whether checkpoint diagnostics can select among existing repair conditions while an independent behavioral gate rejects apparent safety recovery accompanied by over-refusal, utility loss, incomplete coverage, or artifact-integrity failure. The frozen design uses prompt-only cross-source deduplication, fixed LoRA drift recipes, SafeTune repair conditions, CircuitKIT-derived localization only after parity gates, AuditKIT scoring, matched random-location controls, and leave-one-checkpoint-out routing regret. The study does not claim a new edit algorithm or that an edit proves causal safety localization.

The completed evidence is CuratorKIT data curation and one bounded local NF4 feasibility smoke. CuratorKIT produced frozen train/development/validation/sealed splits of 256/60/90/150 records in 74.6 seconds, with 9,212 of 9,219 candidate records surviving the leakage barrier. The harmless smoke produced 3 new tokens from a 378-token chat input, with 2,694,060,544 bytes peak CUDA allocation in 152 seconds; it is not a baseline or evaluation. No drift checkpoint, repair, circuit measurement, router result, sealed score, model artifact, or WSP session exists. H1–H6 are untested. The first RunPod setup failed during NFS root-squash archive extraction and its resources were deleted; no experiment output was lost and no paid resource is active. This repository publishes protocol, implementation, provenance, and the aggregate curation report rather than invented safety metrics.

## Track fit and toolkit use

The intended Track 2b problems are **6** (repairing safety drift without retraining), **9** (locating where safety lives), and **10** (circuit-level interventions). Problem 5 is mentioned only where factually applicable; this write-up does not claim it as an achieved outcome.

| Repository | Status in NeuroForge |
|---|---|
| CuratorKIT | Used for the completed full curation and frozen manifests. |
| SafeTune | Planned M1/M2/M3 repair integration; no repair invocation or result. |
| CircuitKIT | Planned diagnostic/ranking integration; no parity, discovery, or circuit result. |
| AuditKIT | Imported and wired for scoring; no model-scoring result. |
| AlignTune / AgentTune | Inspected, not used. Direct TRL/PEFT is the planned fault-training path. |

The protocol chains CuratorKIT → TRL/PEFT → SafeTune/CircuitKIT → AuditKIT, but only the CuratorKIT stage has completed in a real study run. Source inspection, package import, implemented adapters, and model execution are distinct states.

## Method summary

The model is `CohereLabs/tiny-aya-global` at revision `af89d219b53ed9b13b8a4645f9c8028973510324`; the planned guardian is `ibm-granite/granite-guardian-3.1-2b` at `81145486e85c6c82c01e759c0356d9d6da4d21a5`. XSTest is pinned to `d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d` (CC-BY-4.0) and GSM8K `main` to `740312add88f781978c0658806c59bc2815b9866` (MIT). The model is CC-BY-NC-4.0; this is non-commercial research.

Before splits, curation used exact deduplication and MinHash at threshold 0.85, character 3-grams, 128 permutations, seed 42, comparing prompts with a constant output. The curation manifest SHA-256 is `4e03e4ed07f0f41c0351ad2f1e3788bcffd6d98352cbae0b4d79ccf3ec247041` and its configuration hash is `53c631aad1b9db410a2055d22b0a2c155e7be3c418092f6847ab4211a799b85b`.

Six fixed planned fault recipes (three F1 and three F2) vary rank, learning rate, steps, seed, and answer format; actual checkpoint count is **zero**. Planned conditions are B0, M1, M2, M3, M4, R0/R1/R2, with W1 optional. The acceptance gate requires complete finite safety, over-refusal, and utility metrics; integrity and reload verification; at least 50% recovery of each positive observed drift gap; at most +5 percentage points over-refusal and −5 points utility versus drift. The router is planned to use five diagnostics, fold-local median imputation, and a depth-2 tree. No gate decision or router comparison has been observed.

## Results

| Item | Result |
|---|---|
| Curation | completed: 9,219 inputs; 0 exact drops; 7 near drops; 23 over-limit exclusions; 9,212 survivors |
| Frozen core split counts | train 256; development 60; validation 90; sealed 150 |
| Local NF4 smoke | completed feasibility observation only: 378 input tokens; 3 new tokens; 2,694,060,544-byte peak CUDA allocation; 152 s |
| Baseline / faults / repairs / circuits / router / WSP | not completed; no model metrics; zero observed fault checkpoints |

## Reproducibility, limits, and release state

Use the implemented `neuroforge curate`, `smoke`, `baseline`, `make-faults`, `repair`, `audit`, and `route` commands documented in the repository README. `audit` refuses sealed data; the sealed evaluator is an independent boundary. The local NF4 smoke completed one harmless request under its bounded setup; it does not establish any study result. RunPod attempt 1 failed during NFS root-squash archive extraction and its resources were deleted. A restricted-Podman proof validated `tar --no-same-owner` for the next attempt; no paid resource is active. The retained exposure is $0.70 unsettled reservation plus $0.6300000152550638 historical storage, or $1.3300000152550638 against the $7 operational and $10 absolute ceilings.

No NeuroForge Hugging Face model/dataset repositories or collections are published, so no artifact URLs are supplied. The real source repository is https://github.com/Skrisps26/neuroforge. Original NeuroForge code is Apache-2.0 only; upstream model, data, toolkit, and artifact terms remain in force. AI assistance was used for repository development and documentation; all current result claims are bounded to retained evidence.
