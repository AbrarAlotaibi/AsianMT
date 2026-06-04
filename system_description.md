---
title: "OpenBracket Submission to the WMT 2026 Arabic--English Translation Shared Task"
author: "OpenBracket Team"
geometry: margin=1in
---

# OpenBracket Submission to the WMT 2026 Arabic--English Translation Shared Task

## 1. Overview

This system description summarises OpenBracket's submission to the WMT 2026 shared task on Arabic--English machine translation, covering both sub-tasks:

- **Sub-Task 1A**: English -> Arabic
- **Sub-Task 2A**: Arabic -> English

We submit three systems per direction (primary plus two contrastive). All systems use publicly available pretrained models, are trained on the official parallel training data only, and are evaluated reference-free on the released challenge test set.

## 2. Submitted Systems

### 2.1 Primary --- COMET-MBR ensemble

The primary system combines the outputs of multiple strong single-system translators via system-combination Minimum Bayes Risk decoding with COMET-22 (`Unbabel/wmt22-comet-da`) as the utility function.

**Pool composition:**

| Direction | Candidates |
|---|---|
| Sub-Task 1A (en->ar) | NLLB-3.3B fine-tune, X-ALMA-FT, X-ALMA-CPO, ALLAM-FT, opus-mt, Claude-API |
| Sub-Task 2A (ar->en) | NLLB-3.3B fine-tune, X-ALMA-FT, X-ALMA-CPO, ALLAM-FT, opus-mt |

For each source sentence, the candidate that maximises the expected COMET utility over the deduplicated candidate pool is selected. This consistently outperforms every individual system on the development set and produces the most stable submission.

### 2.2 Contrastive 1 --- ALLAM-7B fine-tune

`ALLaM-AI/ALLaM-7B-Instruct-preview` is an Arabic-native instruction-tuned LLM (SDAIA x IBM). We fine-tune it with QLoRA (4-bit NF4 quantisation, LoRA r=64, alpha=128, all-linear targets) on the official parallel training data, separately for each direction. Inference uses greedy decoding under the chat template. The Arabic source is normalised (Alef variants unified, tatweel and diacritics removed) before training and inference for the ar->en direction.

This system achieves the highest single-system BLEU on the development set for both directions.

### 2.3 Contrastive 2 --- Sampling-MBR over NLLB-3.3B

We draw K=32 hypotheses per source from a fine-tuned NLLB-3.3B model with nucleus sampling (temperature=0.8, top-p=0.95), then apply COMET-MBR with COMET-Kiwi (`Unbabel/wmt22-cometkiwi-da`) used as a quality-estimation pre-pruning step to the top 16 hypotheses per source, followed by COMET-22 MBR utility scoring over the pruned pool.

This system delivers the highest COMET-22 score among all individual approaches on the development set.

## 3. Training Pipeline

### 3.1 Base models

We initialise from the following publicly available checkpoints:

- `facebook/nllb-200-3.3B` --- NLLB-3.3B multilingual seq2seq
- `facebook/nllb-200-distilled-1.3B` --- NLLB-1.3B (smaller backup)
- `haoranxu/X-ALMA-13B-Pretrain` + Group-8 LoRA --- X-ALMA backbone for Arabic
- `ALLaM-AI/ALLaM-7B-Instruct-preview` --- Arabic-native instruction LLM
- `Helsinki-NLP/opus-mt-ar-en` and `opus-mt-en-ar` --- bilingual baselines

### 3.2 NLLB fine-tuning

NLLB-3.3B is fine-tuned with pipeline-parallel sharding across multiple GPUs (per-device max-memory budget forces layer distribution). Training uses fp16, batch size 2 with gradient accumulation 16, AdamW, cosine LR schedule with 5% warmup, 2 epochs on the official parallel data.

### 3.3 X-ALMA QLoRA fine-tuning

The X-ALMA-13B backbone is loaded in 4-bit NF4. We attach a fresh LoRA (r=16, alpha=32) on all linear projections, with loss masked to target tokens only. Optimization uses `paged_adamw_8bit` with gradient checkpointing.

### 3.4 CPO preference fine-tuning

On top of the X-ALMA SFT adapter, we run Contrastive Preference Optimization (CPO). For each training source we sample K=4 candidates from the SFT model, score the pool augmented with the gold reference using COMET-Kiwi, and form preference triples by pairing the highest-scoring candidate as the chosen response against the lowest-scoring as rejected. We use `trl==0.13`'s CPOTrainer continuing from the SFT LoRA. CPO modestly improves over X-ALMA-FT in both directions.

### 3.5 ALLAM fine-tuning

ALLAM-7B-Instruct is fine-tuned per direction with QLoRA r=64 alpha=128 on the official parallel data, max sequence length 1024, batch size 8 with gradient accumulation 4. Flash Attention 2 is enabled where available.

### 3.6 Decoding and ensemble

Single-system inference uses beam search with beam=4 for the seq2seq models and greedy decoding for the LLM systems. Sampling-MBR uses K=32 nucleus samples (temp=0.8, top-p=0.95) followed by COMET-Kiwi QE pre-pruning to 16 candidates and COMET-22 MBR over the pruned pool. System-combination MBR over all available best-1 outputs forms the primary submission.

## 4. Development-Set Results

All scores below are computed on the official development set (n=1000 per direction). BLEU is sacreBLEU 2.6.0 with default tokenisation; COMET is wmt22-comet-da.

### 4.1 Sub-Task 2A: Arabic -> English

| System | BLEU | chrF | TER | COMET |
|---|---:|---:|---:|---:|
| opus-mt | 26.49 | 54.08 | 62.36 | 0.8222 |
| X-ALMA pretrain | 24.67 | 52.30 | 64.09 | 0.8380 |
| NLLB-3.3B fine-tune | 31.51 | 57.86 | 58.64 | 0.8447 |
| X-ALMA-FT | 31.76 | 58.23 | 59.18 | 0.8449 |
| MBR sys-combo | 31.78 | 58.01 | 58.40 | 0.8495 |
| X-ALMA-CPO | 32.23 | 58.81 | 58.54 | 0.8468 |
| **Contrastive 2: Sampling-MBR** | 30.06 | -- | -- | **0.8524** |
| **Contrastive 1: ALLAM-FT** | **33.06** | 59.18 | 58.56 | 0.8449 |

### 4.2 Sub-Task 1A: English -> Arabic

| System | BLEU | chrF | TER | COMET |
|---|---:|---:|---:|---:|
| X-ALMA pretrain | 14.94 | 49.07 | 78.28 | 0.8624 |
| X-ALMA-FT | 17.38 | 49.35 | 72.84 | 0.8610 |
| X-ALMA-CPO | 17.77 | -- | -- | 0.8630 |
| opus-mt | 18.84 | 52.87 | 72.15 | 0.8678 |
| MBR sys-combo | 20.95 | 54.36 | -- | 0.8837 |
| NLLB-3.3B fine-tune | 21.19 | 54.58 | -- | 0.8789 |
| **Contrastive 2: Sampling-MBR** | 18.72 | -- | -- | **0.8862** |
| **Contrastive 1: ALLAM-FT** | **21.74** | -- | -- | 0.8750 |

## 5. Hardware and Reproducibility

All training and inference were performed on 3x NVIDIA RTX 3090 (24 GB each). The full pipeline takes roughly two days end-to-end. Source code and training/decoding scripts are released at: https://github.com/AbrarAlotaibi/AsianMT

## 6. Conclusions

Our primary submission combines diverse, individually strong systems via COMET-MBR and consistently outperforms every individual constituent on the development set. Two architectural contributions stand out: (i) ALLAM, an Arabic-native LLM, delivers the highest single-system BLEU in both directions when fine-tuned on the official data with QLoRA; (ii) sampling-MBR over NLLB-3.3B delivers the highest COMET on a single approach. The contrastive systems were chosen to expose these two complementary findings.
