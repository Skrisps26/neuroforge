# Prior art and positioning

**Status:** pre-execution literature and source-positioning review. It is not a systematic review, proof of novelty, or replication. Claims below describe the cited work or its source audit, not NeuroForge results. Each source must be rechecked before any publication quote.

## Narrow positioning

The proposed contribution is narrowly: **checkpoint-level diagnostics selecting among measured existing repair options under an independently enforced behavioral acceptance gate**. SafeTune already provides method dispatch (`RECOVER_REGISTRY`); dispatch infrastructure is not claimed as novel. NeuroForge does not claim a novel weight-edit algorithm, transferable patches, an exhaustive novelty search, or that intervention success proves a circuit/layer stores safety.

The design intentionally separates (1) behavioral repair, (2) mechanistic diagnostic evidence, (3) size-matched random-location controls, and (4) selection/regret analysis. A single positive repair would not establish localization or a generally reusable patch.

## Sources and relevance

| Source | What it contributes to the study design | Boundary for NeuroForge claims |
|---|---|---|
| **NLSR: Neuron-Level Safety Realignment of Large Language Models Against Harmful Fine-Tuning** ([arXiv:2412.12497](https://arxiv.org/abs/2412.12497)) | Prior neuron-level safety realignment after harmful fine-tuning. | NeuroForge does not relabel its whole q/v matrix restoration as neuron-level realignment or claim novelty over NLSR. |
| **Does Editing Provide Evidence for Localization?** ([arXiv:2502.11447](https://arxiv.org/abs/2502.11447)) | Motivates matched random-location controls: behavioral edit success alone is not evidence that a selected component causally stores the behavior. | M2 versus R0/R1/R2 is a control comparison, not proof of localization even if M2 wins. |
| **SafeGene: Reusable Adapters for Transferable Safety Alignment** ([arXiv:2606.06519](https://arxiv.org/abs/2606.06519)) | Existing reusable/transferable safety-adapter work, with recalibration considerations. | NeuroForge makes no transferable/reusable-patch claim. |
| **TRACE: Trajectory-Based Safety Patch Learning for LLM Post-Training Realignment** ([arXiv:2607.16242](https://arxiv.org/abs/2607.16242)) | Existing offline patch learning across fine-tuning trajectories; not a wholly training-free repair pipeline. | NeuroForge does not claim TRACE-style trajectory training or compare against it without an actual compatible implementation. |
| **Refusal Localizes, the Damage Relocates: Safety Layers Under Few-Sample Fine-Tuning** ([arXiv:2610.00320](https://arxiv.org/abs/2610.00320)) | Counters the assumption that safety damage is stably local: successful local recovery can be bypassed when damage relocates. | The study tests distributed versus concentrated observed drift; it does not assume any fixed “safety layer” is robust. |
| **Safety Direction Adapter: A Training-Free Post-Hoc Safety Patch for Fine-Tuned Open-Weight LLMs** ([arXiv:2611.10056](https://arxiv.org/abs/2611.10056)) | Prior training-free post-hoc direction-adapter work. | M3 is a SafeTune library condition; it is not claimed as a newly invented direction adapter. Reported numbers in that paper, if cited later, remain that paper’s results, not ours. |
| **Restoring General and Improving Domain-Specific Safety in Fine-Tuned LLMs Without Retraining** (SPECTRA, [arXiv:2611.10279](https://arxiv.org/abs/2611.10279)) | Existing post-training restoration/low-rank safety work. | NeuroForge does not call its SafeTune SVD heuristic a SPECTRA reproduction. |
| **Fragile Reasoning: A Mechanistic Analysis of LLM Sensitivity to Meaning-Preserving Perturbations** ([arXiv:2604.01639](https://arxiv.org/abs/2604.01639)) | This identifier is a reasoning-sensitivity paper, not a safety-repair paper. | It must not be cited as establishing a safety-drift taxonomy. Full text needs rechecking before any substantive use. |
| SafetyPulse safety-drift monitor ([report](https://huggingface.co/blog/purvansh01/safetypulse-safety-drift-monitor), source `PxA-Labs/safetypulse` at `d4db9f91d1f434086ca0be7af586bfd998760f34`) | A contemporaneous, unreviewed project. The approved source audit found circuit restore failed while rank-8 whole-model rollback worked in its reported setup, with direction-projection signal stronger than circuit signal. | Its headline and held-out-report numbers are different runs/judges and must never be merged. It is neither an independent replication nor a NeuroForge result. |
| **Sparks of Weight-Space Programming** (Takehi, Teoh, Zhang, Khattab, 2026-11-02; [primary post](https://rikiyat.github.io/blog/weight-space-programming/)) | Conceptual basis for optional weight-space programming, not a prerequisite for the core repair/router study. | WSP remains optional. The library is a separate harness from the paper’s original experiments. No WSP claim is made without bounded real sessions, trusted evaluation, and retained transcripts. |
| Independent WSP audit ([`stawils/wsp-audit`](https://github.com/stawils/wsp-audit), `afb2fe881e96de370a218696e8681489324677b0`, `audit/README.md`) | Challenges stronger “reads/understands weights” interpretations and headline baseline comparisons while retaining evidence that sparse edits can have effects. | Any optional WSP report must emphasize behavioral probes, budgets, edit constraints, rollback, and alignment—not anthropomorphic understanding. |
| Neighbor search set: ProCon ([arXiv:2509.06795](https://arxiv.org/abs/2509.06795)); RefusalGuard ([arXiv:2605.01913](https://arxiv.org/abs/2605.01913)); trait-space checkpoint monitoring ([arXiv:2606.07631](https://arxiv.org/abs/2606.07631)); safety output-routing damage ([arXiv:2609.01455](https://arxiv.org/abs/2609.01455)); SafeSeek ([arXiv:2603.23268](https://arxiv.org/abs/2603.23268)) | Nearby work considered during plan review. | The review did not establish the full requested learned repair-selection pipeline, but that is not proof none exists. No priority claim follows from this shortlist. |

## Methodology implications

1. **Random controls are mandatory.** M2 is compared to R0/R1/R2 that preserve whole-matrix q/v granularity, four-layer cardinality, strength, candidate count, generation settings, and validation protocol. The seeds are preregistered `101`, `102`, `103`; controls are not searched and cherry-picked.
2. **Circuit evidence is conditional.** No circuit-derived repair is interpreted until actual Tiny Aya HF/TransformerLens parity, nondegenerate attribution, and cross-checkpoint patch effects exist. A parity failure blocks circuit claims.
3. **Behavioral acceptance is independent.** A candidate is promoted only through frozen safety, over-refusal, utility, integrity, and coverage conditions. Diagnostic proxies are never final safety metrics.
4. **Selection is not benchmark leakage.** The router sees five development/validation diagnostic features and validation-derived passing labels only; sealed scores/outputs are excluded until all choices are frozen.
5. **No paper-result borrowing.** Published figures, headline claims, and toolkit README compatibility claims are not evidence for Tiny Aya in this configuration. NeuroForge will report only its own saved outputs, hashes, and observed calls.

## Relationship to the selected upstream methods

SafeTune’s pinned `apply_safety_vector_restore` is explicitly a SafeTune-original low-rank task-arithmetic heuristic: it forms the aligned-minus-drifted vector, applies per-matrix truncated SVD, and has a full-delta exception fallback. It is not a faithful reproduction of LoX, RESTA, SPECTRA, or a paper method. M1 therefore uses its documented call with rank 8/alpha 1 and verifies the realized edit before describing it as rank-8.

SafeTune C-ΔΘ operates at matched whole `.weight` matrices filtered by layer/module; it is not neuron-level and M2 does not claim it is. The M3 refusal direction is likewise a library condition. CircuitKIT attribution ranks layer-level attention-head mass; it supplies a selection signal, not a proof that an attention layer is the causal storage location.

## Current evidence boundary

No NeuroForge measurement, repair, comparison, model weight, or WSP session exists. In particular, the documents above do not establish H1–H6, localization, transfer, mechanism, safety improvement, or utility retention. They specify why the study retains negative cases and matched controls if execution becomes possible.
