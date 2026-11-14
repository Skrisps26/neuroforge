# Lexsi rules verification

**Verification timestamp:** `2026-11-14T18:05:07+00:00` (live page read).  
**Primary pages:** [hackathon overview](https://hackathon.lexsi.ai/) and [submission page](https://hackathon.lexsi.ai/submission/). The submission page presently embeds its form; the requirements below are taken from the live overview/rules content. Re-read these pages immediately before publication or submission because rules and dates can change.

## Confirmed schedule

| Milestone | Live rule |
|---|---|
| Initial project submission | **15 November 2026, 23:59 AoE**. AoE is UTC−12, therefore **16 November 2026, 11:59 UTC** and **17:29 IST**. |
| Shortlist notice | Up to 20 teams, by 16 November 2026. |
| Finalist deadline | **22 November 2026, 23:59 AoE**; equivalent **23 November 2026, 11:59 UTC**. Finalists submit a video presentation or slides after proving the thesis. |
| Registration | The overview says registration closed 8 November (after an extension). The user has confirmed solo registration. |

## Track 2b eligibility and project obligations

The live rules require a dense, open-weights Cohere Labs model; NeuroForge’s selected core is the Cohere Labs Tiny Aya model pinned in the preregistration. The study is scoped to Track 2b—AI red teaming, safety, and alignment—with these core-eligible problem statements:

- **6:** repairing safety drift without retraining;
- **8:** inference-time safety steering;
- **9:** locating where safety lives; and
- **10:** circuit-level interventions.

The rules ask entrants to chain at least **two** Track repositories in a real pipeline and state that **three or more is preferred**. The intended core pipeline is CuratorKIT → TRL/PEFT → SafeTune/CircuitKIT → AuditKIT: CuratorKIT, SafeTune, CircuitKIT, and AuditKIT are the Track repositories actually intended for execution. Direct TRL/PEFT is deliberately not falsely described as an AlignTune integration.

Other relevant live requirements:

- projects, models/adapters, logs, and datasets are public;
- before/after evaluation must use more than one metric;
- public, ungated Hugging Face **model** and **dataset** collections are required, and repositories inside them must also be public;
- the preferred write-up is a Hugging Face Article, but a complete GitHub README is an explicit fallback if Articles are unavailable;
- the write-up requirements include configuration/licenses, sources/splits/leak checks/licenses, a harmless over-refusal test set, safety and retention results, before/after comparison, limitations, reproducibility, GitHub/team information, and an AI-use statement;
- a quantized/deployed artifact needs measured results in that deployed/quantized form where applicable.

The planned publication targets—after actual verification, not before—are public collections titled `NeuroForge: Models and Patches` and `NeuroForge: Datasets, Audits and Agent Traces`. A collection name or planned repository is not evidence that it exists.

## Team and submission boundary

Solo teams are permitted by the rules. The user confirmed registration as solo. The user has **not** confirmed Discord membership; the live overview directs participants to Discord for questions/tickets. The user must join Discord themselves if not already a member, and should ask the exact eligibility question about the supplementary Muse editor before citing it in a hackathon claim.

The user alone submits the final project and sends any Discord ticket. This project documentation must not contain the registered email or personal contact data. No final submission has been made.

## License and research boundary

NeuroForge’s own code is Apache-2.0. The selected Cohere Tiny Aya model is CC-BY-NC-4.0, so the study is personal, non-commercial research. SafeTune and CircuitKIT are LSAL v1.2 source-available, not OSI open source; their conditions, notices, non-commercial restrictions, and any organizational acknowledgement requirements remain applicable. CuratorKIT, AuditKIT, datasets, guardian model, and artifacts keep their own terms. Apache-2.0 does not relicense any upstream material.

## Current execution boundary

The public GitHub repository is `https://github.com/Skrisps26/neuroforge`, newly created with no commit at this record. The data environment has confirmed imports for CuratorKIT `0.1.21` and AuditKIT `1.1.12` (including the listed curation/scoring symbols), but no model was loaded or trained. No NeuroForge metrics, weights, uploads, cloud resource, or cloud spend exists. RunPod’s recorded balance remains negative with zero current spend/hour, so no billable resource is authorized by this rules verification.
