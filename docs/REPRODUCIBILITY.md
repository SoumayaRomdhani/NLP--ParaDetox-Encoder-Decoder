# Reproducibility guide

## Environment

Observed: **Google Colab**, **NVIDIA Tesla T4**, **Python 3.13.16**, **PyTorch 2.11.0+cu130**. Random seed: **42**. Compatible dependencies are declared in `requirements.txt`; the notebook has its own install-if-missing setup cell.

## How to rerun

1. Open [`notebooks/ParaDetox.ipynb`](../notebooks/ParaDetox.ipynb) in Google Colab with a GPU runtime.
2. Install dependencies or execute the notebook's first configuration cell.
3. Execute all cells sequentially. The dataset downloads from pinned Hugging Face revision `958b799` and is verified with SHA-256.
4. Train baseline, Attention and Pointer models; load best validation checkpoints.
5. Run validation selection, test evaluation and qualitative diagnostics.

**Time and resources:** full model retraining and evaluation can be expensive; internet access is required for ParaDetox and the pretrained **evaluation-only** models. Checkpoints are **not** committed and will need to be regenerated. Re-execution need not yield identical results due to runtime and GPU nondeterminism.

## Provenance and claims

- Primary artifact: the **executed notebook** (102 code cells; accompanying markdown analyses).
- Recorded test: 1,193 distinct sources.
- No pretrained text-generating model was fine-tuned.
- The 66.81% joint proxy is an automatic metric, not independent human assessment.
- The full human evaluation and multiple random seeds are future work, not completed experiments.

## Data & artifact policy

This repository deliberately excludes the raw dataset, model checkpoints, caches, and credentials. Refer to original upstream dataset and model licenses before redistribution.
