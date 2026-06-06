# Paper Gap-Fill: Verified Answers

Generated from repo contents on 2026-06-06.
Every value below comes from files in the repo; exact sources cited inline.

---

## Metric Configuration (cite once in paper)

- **sacreBLEU signature:** `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`
  (from every `*.dev.json` in `results/`)
- **COMET model:** `Unbabel/wmt22-comet-da`
- **COMET-Kiwi (QE) model:** `Unbabel/wmt22-cometkiwi-da`
  (used in `scripts/mbr_decode.py` for QE pre-pruning and in `score_submissions.sh`)

---

## Task 1 — Data Statistics

### Line counts

Command: `awk 'END{print NR}' <file>` (handles missing trailing newline correctly;
`wc -l` under-counts by 1 when the last line lacks `\n`).

| Split    | File                 | Sentence pairs |
|----------|----------------------|---------------:|
| Train    | `Ar-En/train.{ar,en}`   | **21,000** |
| Dev      | `Ar-En/dev.{ar,en}`     | **1,000**  |
| DevTest  | `Ar-En/devtest.{ar,en}` | **500**    |

The corpus is a single bidirectional ar↔en parallel set shared by both
sub-tasks; there are no separate per-direction splits.

**Dev-set size confirmation:** 1,000 sentence pairs per direction. ✓
(All `*.dev.json` files report `"n": 1000`.)

### Monolingual data

Scripts for monolingual download (`scripts/fetch_mono.py`) and
back-translation assembly (`scripts/build_bt_data.py`) exist in the repo,
but:
- No `Ar-En/mono/` directory is present.
- No `Ar-En/unconstrained/` or back-translation outputs exist.
- No model was trained on augmented data in any log that produced a
  submitted system.

**Conclusion: No monolingual data was used in the submitted systems.**

### External parallel data

`scripts/fetch_parallel.py` can pull OPUS corpora (News-Commentary,
GlobalVoices, TED2020, Tatoeba), but it writes to `Ar-En/unconstrained/`
which does not exist. The unconstrained NLLB fine-tune (`run_unconstrained.sh`)
was an experiment only; its results are not among the submitted systems.

**Conclusion: Only the official WMT 2026 parallel data (21,000 pairs) was
used for training.**

### Domain / source

No explicit domain label is recorded in the repo. Inspecting the first
lines of `train.{ar,en}` and `dev.{ar,en}` shows mixed-domain content:
news articles, sports, politics, economics, cultural commentary — consistent
with a WMT-style news/general-domain parallel corpus.

---

## Task 2 — Complete Results Tables

All values from `results/<system>.<direction>.dev.json`. Values that were
previously `--` in `system_description.md` are marked with ★.

### Sub-Task 2A: Arabic → English (dev, n = 1000)

```latex
\begin{tabular}{lrrrr}
\toprule
System & BLEU & chrF & TER & COMET \\
\midrule
opus-mt                & 26.49 & 54.08 & 62.36 & 0.8222 \\
X-ALMA pretrain        & 24.67 & 52.30 & 64.09 & 0.8380 \\
NLLB-3.3B fine-tune    & 31.51 & 57.86 & 58.64 & 0.8447 \\
X-ALMA-FT              & 31.76 & 58.23 & 59.18 & 0.8449 \\
X-ALMA-CPO             & 32.23 & 58.81\starmark & 58.54\starmark & 0.8468 \\
MBR sys-combo          & 31.78 & 58.01 & 58.40 & 0.8495 \\
Sampling-MBR           & 30.06 & 57.26\starmark & 59.33\starmark & \textbf{0.8524} \\
\textbf{ALLAM-FT}      & \textbf{33.06} & 59.18 & 58.56 & 0.8449 \\
\bottomrule
\end{tabular}
```

Source files for ★ values:
- `results/xalma-cpo.ar-en.dev.json` → chrF 58.81, TER 58.54
- `results/sampling-mbr.ar-en.dev.json` → chrF 57.26, TER 59.33


### Sub-Task 1A: English → Arabic (dev, n = 1000)

```latex
\begin{tabular}{lrrrr}
\toprule
System & BLEU & chrF & TER & COMET \\
\midrule
X-ALMA pretrain        & 14.94 & 49.07 & 78.28 & 0.8624 \\
X-ALMA-FT              & 17.38 & 49.35 & 72.84 & 0.8610 \\
X-ALMA-CPO             & 17.77 & 50.44\starmark & 74.35\starmark & 0.8630 \\
opus-mt                & 18.84 & 52.87 & 72.15 & 0.8678 \\
NLLB-3.3B fine-tune    & 21.19 & 54.58 & 68.28\starmark & 0.8789 \\
MBR sys-combo          & 20.95 & 54.36 & 69.52\starmark & 0.8837 \\
Sampling-MBR           & 18.72 & 52.74\starmark & 71.26\starmark & \textbf{0.8862} \\
\textbf{ALLAM-FT}      & \textbf{21.74} & 54.40\starmark & 69.10\starmark & 0.8750 \\
\bottomrule
\end{tabular}
```

Source files for ★ values:
- `results/xalma-cpo.en-ar.dev.json` → chrF 50.44, TER 74.35
- `results/nllb33b.en-ar.dev.json` → TER 68.28
- `results/mbr.en-ar.dev.json` → TER 69.52
- `results/sampling-mbr.en-ar.dev.json` → chrF 52.74, TER 71.26
- `results/allam-ft.en-ar.dev.json` → chrF 54.40 (stored as 54.4), TER 69.10 (stored as 69.1)

**Note:** `allam-ft.en-ar.dev.json` stores `"chrF": 54.4` and `"TER": 69.1`
(one decimal). All other JSONs use two decimals. The values above use the
exact stored precision (54.40 and 69.10 padded for table alignment only).

---

## Task 3 — Claude-API Decision (en→ar primary pool)

### 3a. Model string / prompt

**No generation script for Claude-API exists in the repo.** The translations
were committed directly as data files in commit `cc4a6bf` ("Claude-API
translations: dev + challenge test (ar-en, en-ar)") with no accompanying
code. The user confirms translations were produced via a regular Claude
chat session.

**⚠ CANNOT VERIFY:** The exact model version (e.g. claude-sonnet-4-6,
claude-opus-4-8, etc.) and the prompt/template used are not recorded
anywhere in the repo. For the paper, you should add a footnote with the
model ID you used.

### 3b. Claude-API dev scores (for reference)

From `results/claude-api.ar-en.dev.json` and `results/claude-api.en-ar.dev.json`:

| Direction | BLEU | chrF | TER | COMET |
|-----------|-----:|-----:|----:|------:|
| ar→en     | 23.52 | 53.17 | 68.04 | 0.8367 |
| en→ar     | 20.45 | 55.45 | 72.16 | 0.8746 |

### 3c. Was Claude-API included in the submitted en→ar primary?

**Yes.** `run_final_mbr.sh` lines 107-115 show the en→ar test pool
explicitly includes `predictions/claude-api.en-ar.test.ar`. The final_mbr.log
confirms:

```
en->ar pool (6):
  predictions/nllb33b.en-ar.test.ar
  predictions/xalma-ft.en-ar.test.ar
  predictions/xalma-cpo.en-ar.test.ar
  predictions/allam-ft.en-ar.test.ar
  predictions/opus.en-ar.test.ar
  predictions/claude-api.en-ar.test.ar
```

For ar→en, Claude-API was **excluded** (5-system pool, no Claude).

### 3d. WITH vs WITHOUT Claude-API recomputation

**⚠ CANNOT COMPUTE.** Two blockers:

1. **Missing hypothesis files.** The individual system dev predictions
   (`predictions/nllb33b.en-ar.dev.ar`, `predictions/xalma-ft.en-ar.dev.ar`,
   `predictions/xalma-cpo.en-ar.dev.ar`, `predictions/allam-ft.en-ar.dev.ar`,
   `predictions/opus.en-ar.dev.ar`) are gitignored and not committed to the
   repo. Only the `claude-api.*` files are in `predictions/`. Without these
   files, MBR cannot be re-run.

2. **No GPU in this environment.** COMET-MBR requires GPU inference
   (COMET-22 model scoring of O(K²) hypothesis pairs per sentence).

### 3e. Important discrepancy to flag

The "MBR sys-combo" row in the paper for **en→ar** (BLEU 20.95 / COMET 0.8837)
comes from `mbr.en-ar.dev.json`, which was computed by `run_mbr.sh` using a
**3-system pool** (xalma-ft, nllb33b, opus only — see `results/mbr.log`).

But the **test** primary was produced by `run_final_mbr.sh` using a
**6-system pool** (+ xalma-cpo, allam-ft, claude-api).

The paper should either:
- Clearly state the dev MBR row uses the 3-system pool, or
- Re-run dev MBR with the same 6-system pool used for test, so the dev
  score is a faithful predictor of test performance.

The same issue exists for **ar→en**: dev MBR was 3-system (xalma-ft, nllb33b,
opus) but test was 5-system (+ xalma-cpo, allam-ft).

### 3f. What you need to do

To produce the WITH/WITHOUT comparison, on your GPU machine:

```bash
# 1. Regenerate dev predictions for the missing systems
#    (or locate them on your training machine — they may still be on disk)

# 2. Run 6-system MBR (WITH Claude-API):
python scripts/mbr_decode.py --src Ar-En/clean/dev.en \
    --candidates predictions/nllb33b.en-ar.dev.ar \
                 predictions/xalma-ft.en-ar.dev.ar \
                 predictions/xalma-cpo.en-ar.dev.ar \
                 predictions/allam-ft.en-ar.dev.ar \
                 predictions/opus.en-ar.dev.ar \
                 predictions/claude-api.en-ar.dev.ar \
    --output predictions/mbr6-with-claude.en-ar.dev.ar
python scripts/evaluate.py --hyp predictions/mbr6-with-claude.en-ar.dev.ar \
    --ref Ar-En/clean/dev.ar --src Ar-En/clean/dev.en --comet

# 3. Run 5-system MBR (WITHOUT Claude-API):
python scripts/mbr_decode.py --src Ar-En/clean/dev.en \
    --candidates predictions/nllb33b.en-ar.dev.ar \
                 predictions/xalma-ft.en-ar.dev.ar \
                 predictions/xalma-cpo.en-ar.dev.ar \
                 predictions/allam-ft.en-ar.dev.ar \
                 predictions/opus.en-ar.dev.ar \
    --output predictions/mbr5-no-claude.en-ar.dev.ar
python scripts/evaluate.py --hyp predictions/mbr5-no-claude.en-ar.dev.ar \
    --ref Ar-En/clean/dev.ar --src Ar-En/clean/dev.en --comet
```

---

## Task 4 — Test / Challenge Set Status

### Challenge test sources

| Direction | Source file | Lines |
|-----------|------------|------:|
| ar→en | `Challenge_Test/Sub-Task 2A_Arabic-to-English/Ar-to-En/test_challenge_ar_ar-to-en.txt` | 1,500 |
| en→ar | `Challenge_Test/Sub-Task 1A_ English-to- Arabic/En-to-Ar/test_challenge_en_en-to-ar.txt` | 1,500 |

### Submitted outputs

All 6 submission files are present (1,500 lines each):

| System | ar→en | en→ar |
|--------|-------|-------|
| Primary (MBR sys-combo) | `results/submission-primary.ar-en.test.en` | `results/submission-primary.en-ar.test.ar` |
| Contrastive 1 (ALLAM-FT) | `results/submission-contrastive1-allam.ar-en.test.en` | `results/submission-contrastive1-allam.en-ar.test.ar` |
| Contrastive 2 (Sampling-MBR) | `results/submission-contrastive2-sampling-mbr.ar-en.test.en` | `results/submission-contrastive2-sampling-mbr.en-ar.test.ar` |

### Reference-free QE scores

`score_submissions.sh` exists and would compute COMET-Kiwi QE scores on
all 6 files, but **`results/submissions_qe.json` does not exist** — the
script was apparently never successfully run (or its output was not committed).

### Official scores

**No official test scores are available in the repo.** The test set is blind
(no reference translations released). Evaluation is performed by the shared
task organizers.

### Claude-API test predictions

`predictions/claude-api.ar-en.test.en` (1,500 lines) and
`predictions/claude-api.en-ar.test.ar` (1,500 lines) exist. The en→ar file
was used in the primary MBR pool; the ar→en file was not.

---

## Summary of gaps that remain

| Item | Status |
|------|--------|
| All chrF/TER cells in both tables | ✅ Filled from cached JSON |
| Claude-API model version & prompt | ❌ Not recorded in repo — add manually |
| WITH vs WITHOUT Claude-API MBR comparison | ❌ Cannot compute (missing dev hyp files + no GPU) — see §3f for commands |
| Dev MBR pool ≠ test MBR pool discrepancy | ⚠ Flag in paper (3-system dev vs 5/6-system test) |
| Test QE scores (COMET-Kiwi) | ❌ Not computed — run `score_submissions.sh` on GPU |
| Official test scores | ❌ Pending shared task evaluation |
