# WMT26 — Arabic-Asian MT (Sub-Tasks 1A & 2A: en↔ar)

Fine-tuning pipeline for the WMT26 Low-Resource Arabic-Asian shared task,
covering both directions between Arabic and English (~21k parallel pairs).

- **Primary system:** `facebook/nllb-200-distilled-600M` fine-tuned bidirectionally.
- **Contrastive system:** Two `Helsinki-NLP/opus-mt-*` bilingual models, one per direction.

Data is gitignored (released only to registered participants).

## Layout

```
data/
  raw/        # Original task files: train|dev|devtest.{ar,en}
  clean/      # Output of scripts/clean.py
scripts/
  clean.py        # Filter train (dedupe, length/ratio caps); passthrough dev/devtest
  train_nllb.py   # Fine-tune NLLB-600M, both directions in one model
  train_opus.py   # Fine-tune one opus-mt direction (run twice)
  translate.py    # Generate translations from a checkpoint
  evaluate.py     # sacreBLEU + chrF + TER (+ optional COMET)
checkpoints/   # (gitignored) training outputs
predictions/   # (gitignored) generated translations
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Run

```bash
# 1. Clean training data (dev/devtest are passed through unchanged)
python scripts/clean.py

# 2a. Primary: NLLB-600M, both directions in one model
python scripts/train_nllb.py --epochs 3

# 2b. Contrastive: opus-mt, one model per direction
python scripts/train_opus.py --direction ar-en --epochs 5
python scripts/train_opus.py --direction en-ar --epochs 5

# 3. Translate devtest with each system
python scripts/translate.py --model checkpoints/nllb-aren/best --kind nllb \
    --direction ar-en --input data/clean/devtest.ar --output predictions/nllb.ar-en.devtest.en
python scripts/translate.py --model checkpoints/nllb-aren/best --kind nllb \
    --direction en-ar --input data/clean/devtest.en --output predictions/nllb.en-ar.devtest.ar
python scripts/translate.py --model checkpoints/opus-ar-en/best --kind opus \
    --direction ar-en --input data/clean/devtest.ar --output predictions/opus.ar-en.devtest.en
python scripts/translate.py --model checkpoints/opus-en-ar/best --kind opus \
    --direction en-ar --input data/clean/devtest.en --output predictions/opus.en-ar.devtest.ar

# 4. Score against references
python scripts/evaluate.py --hyp predictions/nllb.ar-en.devtest.en --ref data/clean/devtest.en
python scripts/evaluate.py --hyp predictions/nllb.en-ar.devtest.ar --ref data/clean/devtest.ar
# add --comet --src ... for COMET (downloads a ~2 GB model)
```

## Hyperparameters (12 GB GPU defaults)

- **NLLB-600M:** `batch=4 × grad_accum=8` (eff. 32), `lr=3e-5`, `epochs=3`, fp16, grad checkpointing
- **opus-mt:** `batch=16 × grad_accum=2` (eff. 32), `lr=5e-5`, `epochs=5`, fp16

Adjust via CLI flags; see `--help` on each script.
