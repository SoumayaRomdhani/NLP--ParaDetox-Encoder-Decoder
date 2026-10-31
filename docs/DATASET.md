# ParaDetox dataset and preprocessing

**Original source:** [s-nlp/paradetox](https://huggingface.co/datasets/s-nlp/paradetox), revision `958b799`.

The notebook downloads a pinned `train.tsv` and checks the SHA-256 digest. Each row pairs `en_toxic_comment` (source) with `en_neutral_comment` (target); a source may have multiple references.

| Partition | Pairs | Distinct normalized sources |
|---|---:|---:|
| Raw dataset | 19,744 | — |
| Cleaned total | 19,003 | — |
| Training (initial) | 15,201 | 9,541 |
| Validation | 1,901 | 1,193 |
| Test | 1,901 | 1,193 |
| Training (effective with augmentation) | 16,823 | contains resampled examples |

**Normalization:** Unicode NFKC, whitespace consolidation and lowercasing; punctuation and negations remain. Empty and duplicate entries are removed.

**Leakage mitigation:** pairs connected through normalized identical source or target text are grouped before an approximately 80/10/10 split. Exact group overlap is checked; semantically similar paraphrases may still appear across partitions.

**Tokenization:** SentencePiece BPE, 2,000 subword vocabulary target, trained only on initial train text. Special tokens `PAD=0, UNK=1, BOS=2, EOS=3`; embedding size 96. Padding is excluded from token-level NLL.

**Train-only augmentation:** 1,216 neutral→neutral identity pairs plus 406 targeted resamples (e.g., numbers and negations). The 406 are not all unique new examples. Validation and test remain unaugmented.

This repository does not redistribute the dataset. Check the original dataset's terms.
