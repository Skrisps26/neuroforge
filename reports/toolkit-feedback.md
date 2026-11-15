# Toolkit feedback: observed integration friction

This note separates observed behavior from source-derived limitations and inference. It reports no model-quality result.

## Observed

- **Tiny Aya chat-template accounting:** a harmless 13-token raw prompt became 378 tokens after the complete chat template. In the installed Transformers version, `apply_chat_template(tokenize=True)` returned a `BatchEncoding`; `len(result)` was not the token count. Token budgets must be measured from the rendered encoding, not raw prompts or container length.
- **CuratorKIT leakage comparison:** the completed 74.6-second curation required a project-side prompt-only comparison representation with constant output (`[NEUROFORGE_PROMPT_COMPARISON]`) to ensure alternate answers could not evade MinHash. The selected MinHash parameters were threshold 0.85, 3-character n-grams, 128 permutations, and seed 42. The aggregate run yielded 7 near-duplicate drops; it did not test model behavior.
- **Combined toolkit import:** after the CUDA wheel installation, a combined toolkit import exceeded a 90-second bound without traceback or success output. That is an observed timeout, not proof of a package defect or incompatibility.
- **Quantized local feasibility:** the RTX 2050 has 4,096 MiB VRAM. A bounded NF4 smoke completed one harmless generation: 378 input tokens, 3 new tokens, 2,694,060,544 bytes peak CUDA allocation, and 152 seconds process time. This permits no claim about baseline generation, training, repair, packed-weight editing, or model quality.

## Source-derived behavior, not a run result

- The pinned SafeTune `apply_safety_vector_restore` source uses truncated SVD for each eligible delta but has a full-delta fallback on SVD exception. NeuroForge must inspect the realized edit/effective rank and must not call a fallback result “rank 8.” No SafeTune repair call has been observed in this study.
- SafeTune C-ΔΘ selects matching weight matrices; it is not a neuron-level intervention. CircuitKIT’s Tiny Aya support is experimental and must pass a real HF/TransformerLens parity gate before circuit-derived claims.
- AuditKIT's Granite Guardian scoring is planned with the pinned local guardian. Its imports/signatures were inspected, but no model response was scored.

## Inference requiring follow-up

Memory pressure is a plausible contributor to the combined-import timeout: the workstation had constrained available RAM/swap during inspection. This is not established causation. Reproduction should record process memory, import order, and timeout outcome before reporting an incompatibility.

## Requests

1. Document template-token accounting examples that distinguish a `BatchEncoding` container from token length.
2. Expose/report an explicit SafeTune fallback/effective-rank field so callers cannot confuse a full-delta fallback with the requested rank.
3. Publish a bounded Tiny Aya compatibility example for CircuitKIT that states parity checks, precision, and memory requirements.
4. Document practical low-VRAM quantization limits and the prohibition on editing packed bitsandbytes weights; recommend merging into a fresh unquantized model for weight-space repairs.
