# Inference and decoding

The decoder starts at `BOS`, forbids outputting `PAD`/`BOS` and terminates on `EOS` or after a maximum of **64 generated tokens**.

- **Greedy:** selects the highest-probability next token at each step.
- **Beam:** explores multiple candidate sequences, with length normalization; variants suppress repeated trigrams.
- **Safe Decoding:** heuristically reranks model-generated candidates for plausibility, completion and preservation of numeric details / selected names.

No reference paraphrase is injected into decoding. The safe reranker is **not** a learned grammatical verifier: some ungrammatical outputs remain.

**Selection:** candidates were compared first on 250 validation sources and then on the complete 1,193-source validation split. The final scores were calculated on 1,193 distinct held-out test sources.
