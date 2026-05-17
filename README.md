# WMT26 — Arabic-Asian MT

Pipeline for the WMT26 Low-Resource Arabic-Asian shared task. Currently set up
for the **Arabic ↔ English** pair (Sub-Tasks 1A and 2A).

- **Primary system:** `facebook/nllb-200-distilled-600M` fine-tuned bidirectionally on the pair.
- **Contrastive system:** Two `Helsinki-NLP/opus-mt-*` bilingual models, one per direction.

## Layout

```
Ar-En/              # Raw task data for the ar<->en pair (~21k train pairs)
  train.ar  train.en
  dev.ar    dev.en
  devtest.ar devtest.en
  clean/            # (gitignored) produced by scripts/clean.py

scripts/
  clean.py        # Dedupe + length/ratio filter for train; passthrough for dev/devtest
  train_nllb.py   # Fine-tune NLLB-600M, both directions in one model
  train_opus.py   # Fine-tune one opus-mt direction (run twice)
  translate.py    # Generate translations from a checkpoint
  evaluate.py     # sacreBLEU + chrF + TER (+ optional COMET)

checkpoints/      # (gitignored) training outputs
predictions/      # (gitignored) generated translations
```

Future Asian pairs would slot in as sibling folders: `Ar-Hi/`, `Ar-Bn/`, `Ar-Id/`, `Ar-Ur/`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Run (Ar↔En)

```bash
# 1. Clean training data (writes Ar-En/clean/)
python scripts/clean.py

# 2a. Primary: NLLB-600M, both directions in one model
python scripts/train_nllb.py --epochs 3

# 2b. Contrastive: opus-mt, one model per direction
python scripts/train_opus.py --direction ar-en --epochs 5
python scripts/train_opus.py --direction en-ar --epochs 5

# 3. Translate devtest with each system
python scripts/translate.py --model checkpoints/nllb-Ar-En/best --kind nllb \
    --direction ar-en --input Ar-En/clean/devtest.ar --output predictions/nllb.ar-en.devtest.en
python scripts/translate.py --model checkpoints/nllb-Ar-En/best --kind nllb \
    --direction en-ar --input Ar-En/clean/devtest.en --output predictions/nllb.en-ar.devtest.ar
python scripts/translate.py --model checkpoints/opus-Ar-En-ar-en/best --kind opus \
    --direction ar-en --input Ar-En/clean/devtest.ar --output predictions/opus.ar-en.devtest.en
python scripts/translate.py --model checkpoints/opus-Ar-En-en-ar/best --kind opus \
    --direction en-ar --input Ar-En/clean/devtest.en --output predictions/opus.en-ar.devtest.ar

# 4. Score against references
python scripts/evaluate.py --hyp predictions/nllb.ar-en.devtest.en --ref Ar-En/devtest.en
python scripts/evaluate.py --hyp predictions/nllb.en-ar.devtest.ar --ref Ar-En/devtest.ar
# add `--comet --src Ar-En/devtest.<src>` for COMET (~2 GB model download)
```

## Hyperparameters (12 GB GPU defaults)

- **NLLB-600M:** `batch=4 × grad_accum=8` (eff. 32), `lr=3e-5`, `epochs=3`, fp16, grad checkpointing
- **opus-mt:** `batch=16 × grad_accum=2` (eff. 32), `lr=5e-5`, `epochs=5`, fp16

Adjust via CLI flags; see `--help` on each script.

## Notes on the data

The WMT26 task data is released only to registered participants. This repo includes
the Ar↔En split because you (the registered participant) chose to track it. If you
make this repo public, the data becomes redistributable — consider keeping the repo
private or removing `Ar-En/*.{ar,en}` before publishing.
