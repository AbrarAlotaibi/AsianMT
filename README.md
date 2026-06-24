# OpenBracket @ WMT 2026 Arabic–English MT

Code, predictions, and results for the **OpenBracket** submission to the
[WMT 2026 Low-Resource Arabic–Asian Machine Translation shared task](https://www2.statmt.org/wmt26/),
covering Sub-Task 1A (English → Arabic) and Sub-Task 2A (Arabic → English).

---

## Submitted Systems

| System | Description |
|--------|-------------|
| **Primary** | COMET-MBR system combination over 5 models |
| **Contrastive 1** | ALLAM-7B fine-tuned with QLoRA |
| **Contrastive 2** | Sampling-MBR over fine-tuned NLLB-3.3B |

The primary system selects, for each source sentence, the candidate that maximises
expected COMET-22 utility over the deduplicated outputs of all five component models
(NLLB-3.3B FT, X-ALMA-FT, X-ALMA-CPO, ALLAM-FT, opus-mt).

See [`system_description.md`](system_description.md) for full details.

---

## Development-Set Results

All scores computed on the official development set (n = 1 000 per direction).
BLEU: sacreBLEU 2.6.0 (signature `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp`).
COMET: `Unbabel/wmt22-comet-da`.

### Sub-Task 2A — Arabic → English

| System | BLEU | chrF | TER | COMET |
|--------|-----:|-----:|----:|------:|
| opus-mt | 26.49 | 54.08 | 62.36 | 0.8222 |
| X-ALMA pretrain | 24.67 | 52.30 | 64.09 | 0.8380 |
| NLLB-3.3B fine-tune | 31.51 | 57.86 | 58.64 | 0.8447 |
| X-ALMA-FT | 31.76 | 58.23 | 59.18 | 0.8449 |
| X-ALMA-CPO | 32.23 | 58.81 | 58.54 | 0.8468 |
| Contrastive 1: ALLAM-FT | **33.06** | 59.18 | 58.56 | 0.8449 |
| Contrastive 2: Sampling-MBR | 30.06 | 57.26 | 59.33 | **0.8524** |
| **Primary: MBR sys-combo** | 32.97 | **59.10** | **57.32** | 0.8531 |

### Sub-Task 1A — English → Arabic

| System | BLEU | chrF | TER | COMET |
|--------|-----:|-----:|----:|------:|
| X-ALMA pretrain | 14.94 | 49.07 | 78.28 | 0.8624 |
| X-ALMA-FT | 17.38 | 49.35 | 72.84 | 0.8610 |
| X-ALMA-CPO | 17.77 | 50.44 | 74.35 | 0.8630 |
| opus-mt | 18.84 | 52.87 | 72.15 | 0.8678 |
| NLLB-3.3B fine-tune | 21.19 | 54.58 | 68.28 | 0.8789 |
| Contrastive 2: Sampling-MBR | 18.72 | 52.74 | 71.26 | **0.8862** |
| **Primary: MBR sys-combo** | 21.48 | 54.90 | 68.97 | 0.8865 |
| Contrastive 1: ALLAM-FT | **21.74** | 54.40 | 69.10 | 0.8750 |

---

## Repository Layout

```
Ar-En/                   # Official WMT 2026 parallel data (ar↔en)
  train.{ar,en}          #   21 000 sentence pairs
  dev.{ar,en}            #    1 000 sentence pairs
  devtest.{ar,en}        #      500 sentence pairs

Challenge_Test/          # Blind challenge test inputs (1 500 sentences each direction)

scripts/
  clean.py               # Deduplication and length/ratio filtering
  fetch_parallel.py      # Download additional parallel corpora
  fetch_mono.py          # Download monolingual data
  filter_training_data.py
  normalize_arabic.py    # Alef/tatweel/diacritic normalisation

  train_nllb.py          # Fine-tune NLLB-3.3B (seq2seq, pipeline-parallel)
  train_opus.py          # Fine-tune Helsinki-NLP/opus-mt-*
  xalma_finetune.py      # QLoRA fine-tune X-ALMA-13B (SFT)
  cpo_finetune.py        # CPO preference fine-tuning on top of X-ALMA SFT
  allam_finetune.py      # QLoRA fine-tune ALLAM-7B-Instruct

  translate.py           # Beam-search inference for seq2seq models
  xalma_translate.py     # Inference for X-ALMA
  allam_translate.py     # Inference for ALLAM
  contradecode.py        # Contrastive / penalty-based decoding utilities

  build_bt_data.py       # Back-translation data construction
  build_preference_data.py  # COMET-ranked preference pairs for CPO
  mbr_decode.py          # MBR decoding (sampling-MBR and system-combination MBR)

  parallel_dl_comet.py   # Download COMET / COMET-Kiwi model weights
  parallel_dl_hf.py      # Parallel HuggingFace checkpoint downloads
  evaluate.py            # sacreBLEU + chrF + TER (+ optional COMET)

predictions/             # Per-system best-1 outputs on dev and test
results/
  *.json                 # Dev-set scores (BLEU/chrF/TER/COMET) per system
  submission-primary.*   # Final primary submission (MBR sys-combo)
  submission-contrastive1-allam.*
  submission-contrastive2-sampling-mbr.*

environment.yml          # Conda environment (CUDA 12.1)
requirements.txt         # pip dependencies
system_description.md    # Full system description paper
```

---

## Setup

**Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate asianmt
```

For CUDA 11.8 instead of 12.1, change `pytorch-cuda=11.8` in `environment.yml`.

**pip:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Reproducing Results

### 1. Prepare data
```bash
python scripts/clean.py --src Ar-En/train.ar --tgt Ar-En/train.en --out Ar-En/clean/
python scripts/normalize_arabic.py --input Ar-En/clean/train.ar --output Ar-En/clean/train.norm.ar
```

### 2. Fine-tune component models
```bash
# NLLB-3.3B (both directions, pipeline-parallel across GPUs)
python scripts/train_nllb.py --src-lang ara_Arab --tgt-lang eng_Latn --epochs 2

# ALLAM-7B (one direction at a time)
python scripts/allam_finetune.py --direction ar-en
python scripts/allam_finetune.py --direction en-ar

# X-ALMA SFT then CPO
python scripts/xalma_finetune.py --direction ar-en
python scripts/cpo_finetune.py  --direction ar-en
```

### 3. Generate translations
```bash
python scripts/translate.py      --model checkpoints/nllb-3.3b-ar-en --direction ar-en \
    --input Ar-En/dev.ar --output predictions/nllb33b.ar-en.dev.en
python scripts/allam_translate.py --direction ar-en \
    --input Ar-En/dev.ar --output predictions/allam-ft.ar-en.dev.en
# ... (repeat per model and direction)
```

### 4. MBR system combination (primary)
```bash
python scripts/mbr_decode.py \
    --candidates predictions/nllb33b.ar-en.dev.en \
                 predictions/xalma-ft.ar-en.dev.en \
                 predictions/xalma-cpo.ar-en.dev.en \
                 predictions/allam-ft.ar-en.dev.en \
                 predictions/opus.ar-en.dev.en \
    --output predictions/final-mbr.ar-en.dev.en
```

### 5. Evaluate
```bash
python scripts/evaluate.py \
    --hyp predictions/final-mbr.ar-en.dev.en \
    --ref Ar-En/dev.en \
    --comet --src Ar-En/dev.ar
```

---

## Hardware

All experiments ran on 3× NVIDIA RTX 3090 (24 GB each). Full pipeline runtime is
approximately two days end-to-end.

---

## Citation

If you use this code or the released predictions, please cite our system description paper
(citation to be added upon proceedings publication).
