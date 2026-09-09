# Camera-Ready Experiments: MBR Pool Ablation and Significance Tests

Generated 2026-09-05 by `scripts/camera_ready_report.py` from `results/camera_ready/`. Official DEV set, n=1000 per direction. Every number here is read from a result file written by the commands in Section 4; nothing is filled in by hand.

## 0. Status

All input files present.

To produce every table below: `CUDA_VISIBLE_DEVICES=0 bash run_camera_ready.sh` (GPU; COMET-22 MBR over 12 pools + COMET scoring of 12 outputs + 8 sentence-level COMET passes, roughly 30-45 min on one RTX 3090), then re-run `python scripts/camera_ready_report.py`.

## 1. Setup

- Pool (both directions): NLLB-3.3B FT, X-ALMA-FT, X-ALMA-CPO, ALLaM-FT, opus-mt (`nllb33b`, `xalma-ft`, `xalma-cpo`, `allam-ft`, `opus`).
- MBR: `scripts/mbr_decode.py`, utility `Unbabel/wmt22-comet-da`, no QE pruning, per-sentence dedup of identical candidates, expected utility over the pool (self-pair included).
- Metrics: sacreBLEU 2.6.0 defaults via `scripts/evaluate.py`; signature `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`; COMET `Unbabel/wmt22-comet-da`.
- References: raw `Ar-En/dev.{ar,en}` (the recorded per-system dev scores reproduce exactly against these files with sacreBLEU 2.6.0).
- Paired bootstrap: 1000 resamples with replacement, sacreBLEU default seed 12345, p-value as in sacreBLEU (`(#{shifted |delta_b| > |delta|} + 1) / (N + 1)`); the COMET test applies the identical procedure to sentence-level wmt22-comet-da scores (`scripts/paired_bootstrap_comet.py`).

## 2. Task A -- Leave-one-out ablation of the MBR pool

### Sub-Task 2A: Arabic -> English (dev, n=1000)

Recorded full-pool score in the paper (`results/mbr5pub.ar-en.dev.json`): BLEU 32.97 / chrF 59.10 / TER 57.32 / COMET 0.8531.
Re-generated full pool **matches** the recorded row exactly.

```latex
\begin{tabular}{lrrrrrr}
\toprule
Pool removed & BLEU & chrF & TER & COMET & $\Delta$BLEU & $\Delta$COMET \\
\midrule
none (full 5-system pool) & 32.97 & 59.10 & 57.32 & 0.8531 & +0.00 & +0.0000 \\
$-$\,NLLB-3.3B FT & 32.91 & 59.24 & 57.50 & 0.8519 & -0.06 & -0.0012 \\
$-$\,X-ALMA-FT & 32.91 & 58.99 & 57.09 & 0.8527 & -0.06 & -0.0004 \\
$-$\,X-ALMA-CPO & 32.61 & 58.78 & 57.35 & 0.8527 & -0.36 & -0.0004 \\
$-$\,ALLaM-FT & 32.56 & 58.89 & 57.69 & 0.8512 & -0.41 & -0.0019 \\
$-$\,opus-mt & 33.09 & 59.36 & 57.35 & 0.8532 & +0.12 & +0.0001 \\
\bottomrule
\end{tabular}
```

Largest COMET loss when removing **ALLaM-FT** (-0.0019); smallest when removing **opus-mt** (+0.0001).

Hypotheses: `predictions/mbr5pub.ar-en.dev.en` (full pool), `predictions/camera_ready/mbr-minus-<sys>.ar-en.dev.en` (leave-one-out).

### Sub-Task 1A: English -> Arabic (dev, n=1000)

Recorded full-pool score in the paper (`results/mbr5pub.en-ar.dev.json`): BLEU 21.48 / chrF 54.90 / TER 68.97 / COMET 0.8865.
Re-generated full pool **matches** the recorded row exactly.

```latex
\begin{tabular}{lrrrrrr}
\toprule
Pool removed & BLEU & chrF & TER & COMET & $\Delta$BLEU & $\Delta$COMET \\
\midrule
none (full 5-system pool) & 21.48 & 54.90 & 68.97 & 0.8865 & +0.00 & +0.0000 \\
$-$\,NLLB-3.3B FT & 20.42 & 53.67 & 70.65 & 0.8835 & -1.06 & -0.0030 \\
$-$\,X-ALMA-FT & 21.52 & 55.10 & 69.08 & 0.8863 & +0.04 & -0.0002 \\
$-$\,X-ALMA-CPO & 21.65 & 55.07 & 68.73 & 0.8865 & +0.17 & +0.0000 \\
$-$\,ALLaM-FT & 20.45 & 53.74 & 70.09 & 0.8831 & -1.03 & -0.0034 \\
$-$\,opus-mt & 21.10 & 54.29 & 69.17 & 0.8852 & -0.38 & -0.0013 \\
\bottomrule
\end{tabular}
```

Largest COMET loss when removing **ALLaM-FT** (-0.0034); smallest when removing **X-ALMA-CPO** (+0.0000).

Hypotheses: `predictions/mbr5pub.en-ar.dev.ar` (full pool), `predictions/camera_ready/mbr-minus-<sys>.en-ar.dev.ar` (leave-one-out).

## 3. Task B -- Paired bootstrap significance vs. the primary

### Sub-Task 2A: Arabic -> English (dev, n=1000)

Baseline = full-pool MBR primary. delta = system $-$ primary (so a negative delta on BLEU/chrF/COMET, or positive on TER, means the primary is better).

| Comparison | Metric | primary | system | delta | p | sig. |
|---|---|---:|---:|---:|---:|:---:|
| primary vs ALLaM-FT | BLEU | 32.97 | 33.06 | +0.09 | 0.2947 | **n.s.** |
| primary vs ALLaM-FT | chrF | 59.10 | 59.18 | +0.08 | 0.2847 | **n.s.** |
| primary vs ALLaM-FT | TER | 57.32 | 58.56 | +1.24 | 0.0020 | * |
| primary vs ALLaM-FT | COMET | 0.8531 | 0.8449 | -0.0082 | 0.0010 | * |
| primary vs X-ALMA-CPO | BLEU | 32.97 | 32.23 | -0.74 | 0.0060 | * |
| primary vs X-ALMA-CPO | chrF | 59.10 | 58.81 | -0.29 | 0.0390 | * |
| primary vs X-ALMA-CPO | TER | 57.32 | 58.54 | +1.23 | 0.0010 | * |
| primary vs X-ALMA-CPO | COMET | 0.8531 | 0.8468 | -0.0063 | 0.0010 | * |
| primary vs NLLB-3.3B FT | BLEU | 32.97 | 31.51 | -1.46 | 0.0010 | * |
| primary vs NLLB-3.3B FT | chrF | 59.10 | 57.86 | -1.24 | 0.0010 | * |
| primary vs NLLB-3.3B FT | TER | 57.32 | 58.64 | +1.32 | 0.0010 | * |
| primary vs NLLB-3.3B FT | COMET | 0.8531 | 0.8447 | -0.0084 | 0.0010 | * |
| primary vs Sampling-MBR | BLEU | 32.97 | 30.06 | -2.91 | 0.0010 | * |
| primary vs Sampling-MBR | chrF | 59.10 | 57.26 | -1.84 | 0.0010 | * |
| primary vs Sampling-MBR | TER | 57.32 | 59.33 | +2.01 | 0.0010 | * |
| primary vs Sampling-MBR | COMET | 0.8531 | 0.8524 | -0.0007 | 0.1009 | **n.s.** |

```latex
\begin{tabular}{llrr}
\toprule
Comparison & Metric & $\Delta$ & $p$ \\
\midrule
primary vs.\ ALLaM-FT & BLEU & +0.09 & 0.295$^{\dagger}$ \\
 & chrF & +0.08 & 0.285$^{\dagger}$ \\
 & TER & +1.24 & 0.002 \\
 & COMET & -0.0082 & 0.001 \\
\midrule
primary vs.\ X-ALMA-CPO & BLEU & -0.74 & 0.006 \\
 & chrF & -0.29 & 0.039 \\
 & TER & +1.23 & 0.001 \\
 & COMET & -0.0063 & 0.001 \\
\midrule
primary vs.\ NLLB-3.3B FT & BLEU & -1.46 & 0.001 \\
 & chrF & -1.24 & 0.001 \\
 & TER & +1.32 & 0.001 \\
 & COMET & -0.0084 & 0.001 \\
\midrule
primary vs.\ Sampling-MBR & BLEU & -2.91 & 0.001 \\
 & chrF & -1.84 & 0.001 \\
 & TER & +2.01 & 0.001 \\
 & COMET & -0.0007 & 0.101$^{\dagger}$ \\
\bottomrule
\end{tabular}
```

$^{\dagger}$ = not significant at $p<0.05$.

sacreBLEU signatures (paired bootstrap):

- BLEU: `nrefs:1|bs:1000|seed:12345|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`
- chrF: `nrefs:1|bs:1000|seed:12345|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`
- TER: `nrefs:1|bs:1000|seed:12345|case:lc|tok:tercom|norm:no|punct:yes|asian:no|version:2.6.0`

COMET bootstrap: `Unbabel/wmt22-comet-da`, 1000 resamples, seed 12345, sentence scores cached in `results/camera_ready/comet-sentence-scores.ar-en.json`.

### Sub-Task 1A: English -> Arabic (dev, n=1000)

Baseline = full-pool MBR primary. delta = system $-$ primary (so a negative delta on BLEU/chrF/COMET, or positive on TER, means the primary is better).

| Comparison | Metric | primary | system | delta | p | sig. |
|---|---|---:|---:|---:|---:|:---:|
| primary vs ALLaM-FT | BLEU | 21.48 | 21.74 | +0.26 | 0.1758 | **n.s.** |
| primary vs ALLaM-FT | chrF | 54.90 | 54.40 | -0.49 | 0.0170 | * |
| primary vs ALLaM-FT | TER | 68.97 | 69.10 | +0.14 | 0.2567 | **n.s.** |
| primary vs ALLaM-FT | COMET | 0.8865 | 0.8750 | -0.0115 | 0.0010 | * |
| primary vs X-ALMA-CPO | BLEU | 21.48 | 17.77 | -3.72 | 0.0010 | * |
| primary vs X-ALMA-CPO | chrF | 54.90 | 50.44 | -4.46 | 0.0010 | * |
| primary vs X-ALMA-CPO | TER | 68.97 | 74.35 | +5.38 | 0.0010 | * |
| primary vs X-ALMA-CPO | COMET | 0.8865 | 0.8630 | -0.0235 | 0.0010 | * |
| primary vs NLLB-3.3B FT | BLEU | 21.48 | 21.19 | -0.30 | 0.0999 | **n.s.** |
| primary vs NLLB-3.3B FT | chrF | 54.90 | 54.58 | -0.31 | 0.0390 | * |
| primary vs NLLB-3.3B FT | TER | 68.97 | 68.28 | -0.69 | 0.0160 | * |
| primary vs NLLB-3.3B FT | COMET | 0.8865 | 0.8789 | -0.0076 | 0.0010 | * |
| primary vs Sampling-MBR | BLEU | 21.48 | 18.72 | -2.76 | 0.0010 | * |
| primary vs Sampling-MBR | chrF | 54.90 | 52.74 | -2.15 | 0.0010 | * |
| primary vs Sampling-MBR | TER | 68.97 | 71.26 | +2.29 | 0.0010 | * |
| primary vs Sampling-MBR | COMET | 0.8865 | 0.8862 | -0.0002 | 0.2418 | **n.s.** |

```latex
\begin{tabular}{llrr}
\toprule
Comparison & Metric & $\Delta$ & $p$ \\
\midrule
primary vs.\ ALLaM-FT & BLEU & +0.26 & 0.176$^{\dagger}$ \\
 & chrF & -0.49 & 0.017 \\
 & TER & +0.14 & 0.257$^{\dagger}$ \\
 & COMET & -0.0115 & 0.001 \\
\midrule
primary vs.\ X-ALMA-CPO & BLEU & -3.72 & 0.001 \\
 & chrF & -4.46 & 0.001 \\
 & TER & +5.38 & 0.001 \\
 & COMET & -0.0235 & 0.001 \\
\midrule
primary vs.\ NLLB-3.3B FT & BLEU & -0.30 & 0.100$^{\dagger}$ \\
 & chrF & -0.31 & 0.039 \\
 & TER & -0.69 & 0.016 \\
 & COMET & -0.0076 & 0.001 \\
\midrule
primary vs.\ Sampling-MBR & BLEU & -2.76 & 0.001 \\
 & chrF & -2.15 & 0.001 \\
 & TER & +2.29 & 0.001 \\
 & COMET & -0.0002 & 0.242$^{\dagger}$ \\
\bottomrule
\end{tabular}
```

$^{\dagger}$ = not significant at $p<0.05$.

sacreBLEU signatures (paired bootstrap):

- BLEU: `nrefs:1|bs:1000|seed:12345|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`
- chrF: `nrefs:1|bs:1000|seed:12345|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`
- TER: `nrefs:1|bs:1000|seed:12345|case:lc|tok:tercom|norm:no|punct:yes|asian:no|version:2.6.0`

COMET bootstrap: `Unbabel/wmt22-comet-da`, 1000 resamples, seed 12345, sentence scores cached in `results/camera_ready/comet-sentence-scores.en-ar.json`.

### Comparisons with p >= 0.05

- ar-en: primary vs ALLaM-FT on BLEU (delta +0.09, p = 0.2947)
- ar-en: primary vs ALLaM-FT on chrF (delta +0.08, p = 0.2847)
- ar-en: primary vs Sampling-MBR on COMET (delta -0.0007, p = 0.1009)
- en-ar: primary vs ALLaM-FT on BLEU (delta +0.26, p = 0.1758)
- en-ar: primary vs ALLaM-FT on TER (delta +0.14, p = 0.2567)
- en-ar: primary vs NLLB-3.3B FT on BLEU (delta -0.30, p = 0.0999)
- en-ar: primary vs Sampling-MBR on COMET (delta -0.0002, p = 0.2418)

## 4. Exact commands

All paths are relative to the repository root. `run_camera_ready.sh` runs exactly these for both directions and then this report script.

#### ar-en

```bash
# TASK A -- full pool (reference row)
python scripts/mbr_decode.py --src Ar-En/dev.ar \
    --candidates predictions/nllb33b.ar-en.dev.en predictions/xalma-ft.ar-en.dev.en predictions/xalma-cpo.ar-en.dev.en predictions/allam-ft.ar-en.dev.en predictions/opus.ar-en.dev.en \
    --output predictions/mbr5pub.ar-en.dev.en
python scripts/evaluate.py --hyp predictions/mbr5pub.ar-en.dev.en --ref Ar-En/dev.en --src Ar-En/dev.ar --comet \
    > results/camera_ready/loo-none.ar-en.dev.json

# TASK A -- pool without nllb33b
python scripts/mbr_decode.py --src Ar-En/dev.ar \
    --candidates predictions/xalma-ft.ar-en.dev.en predictions/xalma-cpo.ar-en.dev.en predictions/allam-ft.ar-en.dev.en predictions/opus.ar-en.dev.en \
    --output predictions/camera_ready/mbr-minus-nllb33b.ar-en.dev.en
python scripts/evaluate.py --hyp predictions/camera_ready/mbr-minus-nllb33b.ar-en.dev.en --ref Ar-En/dev.en --src Ar-En/dev.ar --comet \
    > results/camera_ready/loo-minus-nllb33b.ar-en.dev.json

# TASK A -- pool without xalma-ft
python scripts/mbr_decode.py --src Ar-En/dev.ar \
    --candidates predictions/nllb33b.ar-en.dev.en predictions/xalma-cpo.ar-en.dev.en predictions/allam-ft.ar-en.dev.en predictions/opus.ar-en.dev.en \
    --output predictions/camera_ready/mbr-minus-xalma-ft.ar-en.dev.en
python scripts/evaluate.py --hyp predictions/camera_ready/mbr-minus-xalma-ft.ar-en.dev.en --ref Ar-En/dev.en --src Ar-En/dev.ar --comet \
    > results/camera_ready/loo-minus-xalma-ft.ar-en.dev.json

# TASK A -- pool without xalma-cpo
python scripts/mbr_decode.py --src Ar-En/dev.ar \
    --candidates predictions/nllb33b.ar-en.dev.en predictions/xalma-ft.ar-en.dev.en predictions/allam-ft.ar-en.dev.en predictions/opus.ar-en.dev.en \
    --output predictions/camera_ready/mbr-minus-xalma-cpo.ar-en.dev.en
python scripts/evaluate.py --hyp predictions/camera_ready/mbr-minus-xalma-cpo.ar-en.dev.en --ref Ar-En/dev.en --src Ar-En/dev.ar --comet \
    > results/camera_ready/loo-minus-xalma-cpo.ar-en.dev.json

# TASK A -- pool without allam-ft
python scripts/mbr_decode.py --src Ar-En/dev.ar \
    --candidates predictions/nllb33b.ar-en.dev.en predictions/xalma-ft.ar-en.dev.en predictions/xalma-cpo.ar-en.dev.en predictions/opus.ar-en.dev.en \
    --output predictions/camera_ready/mbr-minus-allam-ft.ar-en.dev.en
python scripts/evaluate.py --hyp predictions/camera_ready/mbr-minus-allam-ft.ar-en.dev.en --ref Ar-En/dev.en --src Ar-En/dev.ar --comet \
    > results/camera_ready/loo-minus-allam-ft.ar-en.dev.json

# TASK A -- pool without opus
python scripts/mbr_decode.py --src Ar-En/dev.ar \
    --candidates predictions/nllb33b.ar-en.dev.en predictions/xalma-ft.ar-en.dev.en predictions/xalma-cpo.ar-en.dev.en predictions/allam-ft.ar-en.dev.en \
    --output predictions/camera_ready/mbr-minus-opus.ar-en.dev.en
python scripts/evaluate.py --hyp predictions/camera_ready/mbr-minus-opus.ar-en.dev.en --ref Ar-En/dev.en --src Ar-En/dev.ar --comet \
    > results/camera_ready/loo-minus-opus.ar-en.dev.json

# TASK B -- BLEU/chrF/TER paired bootstrap (1000 resamples, default seed 12345)
sacrebleu Ar-En/dev.en -i predictions/mbr5pub.ar-en.dev.en \
    predictions/allam-ft.ar-en.dev.en \
    predictions/xalma-cpo.ar-en.dev.en \
    predictions/nllb33b.ar-en.dev.en \
    predictions/sampling-mbr.ar-en.dev.en \
    -m bleu chrf ter --paired-bs --paired-bs-n 1000 -f text > results/camera_ready/paired-bs.ar-en.txt
# same test through the sacreBLEU API, full-precision JSON:
python scripts/paired_bootstrap_sacrebleu.py --ref Ar-En/dev.en --baseline predictions/mbr5pub.ar-en.dev.en \
    --systems predictions/allam-ft.ar-en.dev.en predictions/xalma-cpo.ar-en.dev.en predictions/nllb33b.ar-en.dev.en predictions/sampling-mbr.ar-en.dev.en \
    --names allam-ft xalma-cpo nllb33b sampling-mbr --n-resamples 1000 --out results/camera_ready/paired-bs.ar-en.json

# TASK B -- COMET paired bootstrap (sentence-level wmt22-comet-da, 1000 resamples, seed 12345)
python scripts/paired_bootstrap_comet.py --src Ar-En/dev.ar --ref Ar-En/dev.en --baseline predictions/mbr5pub.ar-en.dev.en \
    --systems predictions/allam-ft.ar-en.dev.en predictions/xalma-cpo.ar-en.dev.en predictions/nllb33b.ar-en.dev.en predictions/sampling-mbr.ar-en.dev.en \
    --names allam-ft xalma-cpo nllb33b sampling-mbr --n-resamples 1000 --seed 12345 \
    --sentence-scores results/camera_ready/comet-sentence-scores.ar-en.json \
    --out results/camera_ready/paired-bs-comet.ar-en.json
```

#### en-ar

```bash
# TASK A -- full pool (reference row)
python scripts/mbr_decode.py --src Ar-En/dev.en \
    --candidates predictions/nllb33b.en-ar.dev.ar predictions/xalma-ft.en-ar.dev.ar predictions/xalma-cpo.en-ar.dev.ar predictions/allam-ft.en-ar.dev.ar predictions/opus.en-ar.dev.ar \
    --output predictions/mbr5pub.en-ar.dev.ar
python scripts/evaluate.py --hyp predictions/mbr5pub.en-ar.dev.ar --ref Ar-En/dev.ar --src Ar-En/dev.en --comet \
    > results/camera_ready/loo-none.en-ar.dev.json

# TASK A -- pool without nllb33b
python scripts/mbr_decode.py --src Ar-En/dev.en \
    --candidates predictions/xalma-ft.en-ar.dev.ar predictions/xalma-cpo.en-ar.dev.ar predictions/allam-ft.en-ar.dev.ar predictions/opus.en-ar.dev.ar \
    --output predictions/camera_ready/mbr-minus-nllb33b.en-ar.dev.ar
python scripts/evaluate.py --hyp predictions/camera_ready/mbr-minus-nllb33b.en-ar.dev.ar --ref Ar-En/dev.ar --src Ar-En/dev.en --comet \
    > results/camera_ready/loo-minus-nllb33b.en-ar.dev.json

# TASK A -- pool without xalma-ft
python scripts/mbr_decode.py --src Ar-En/dev.en \
    --candidates predictions/nllb33b.en-ar.dev.ar predictions/xalma-cpo.en-ar.dev.ar predictions/allam-ft.en-ar.dev.ar predictions/opus.en-ar.dev.ar \
    --output predictions/camera_ready/mbr-minus-xalma-ft.en-ar.dev.ar
python scripts/evaluate.py --hyp predictions/camera_ready/mbr-minus-xalma-ft.en-ar.dev.ar --ref Ar-En/dev.ar --src Ar-En/dev.en --comet \
    > results/camera_ready/loo-minus-xalma-ft.en-ar.dev.json

# TASK A -- pool without xalma-cpo
python scripts/mbr_decode.py --src Ar-En/dev.en \
    --candidates predictions/nllb33b.en-ar.dev.ar predictions/xalma-ft.en-ar.dev.ar predictions/allam-ft.en-ar.dev.ar predictions/opus.en-ar.dev.ar \
    --output predictions/camera_ready/mbr-minus-xalma-cpo.en-ar.dev.ar
python scripts/evaluate.py --hyp predictions/camera_ready/mbr-minus-xalma-cpo.en-ar.dev.ar --ref Ar-En/dev.ar --src Ar-En/dev.en --comet \
    > results/camera_ready/loo-minus-xalma-cpo.en-ar.dev.json

# TASK A -- pool without allam-ft
python scripts/mbr_decode.py --src Ar-En/dev.en \
    --candidates predictions/nllb33b.en-ar.dev.ar predictions/xalma-ft.en-ar.dev.ar predictions/xalma-cpo.en-ar.dev.ar predictions/opus.en-ar.dev.ar \
    --output predictions/camera_ready/mbr-minus-allam-ft.en-ar.dev.ar
python scripts/evaluate.py --hyp predictions/camera_ready/mbr-minus-allam-ft.en-ar.dev.ar --ref Ar-En/dev.ar --src Ar-En/dev.en --comet \
    > results/camera_ready/loo-minus-allam-ft.en-ar.dev.json

# TASK A -- pool without opus
python scripts/mbr_decode.py --src Ar-En/dev.en \
    --candidates predictions/nllb33b.en-ar.dev.ar predictions/xalma-ft.en-ar.dev.ar predictions/xalma-cpo.en-ar.dev.ar predictions/allam-ft.en-ar.dev.ar \
    --output predictions/camera_ready/mbr-minus-opus.en-ar.dev.ar
python scripts/evaluate.py --hyp predictions/camera_ready/mbr-minus-opus.en-ar.dev.ar --ref Ar-En/dev.ar --src Ar-En/dev.en --comet \
    > results/camera_ready/loo-minus-opus.en-ar.dev.json

# TASK B -- BLEU/chrF/TER paired bootstrap (1000 resamples, default seed 12345)
sacrebleu Ar-En/dev.ar -i predictions/mbr5pub.en-ar.dev.ar \
    predictions/allam-ft.en-ar.dev.ar \
    predictions/xalma-cpo.en-ar.dev.ar \
    predictions/nllb33b.en-ar.dev.ar \
    predictions/sampling-mbr.en-ar.dev.ar \
    -m bleu chrf ter --paired-bs --paired-bs-n 1000 -f text > results/camera_ready/paired-bs.en-ar.txt
# same test through the sacreBLEU API, full-precision JSON:
python scripts/paired_bootstrap_sacrebleu.py --ref Ar-En/dev.ar --baseline predictions/mbr5pub.en-ar.dev.ar \
    --systems predictions/allam-ft.en-ar.dev.ar predictions/xalma-cpo.en-ar.dev.ar predictions/nllb33b.en-ar.dev.ar predictions/sampling-mbr.en-ar.dev.ar \
    --names allam-ft xalma-cpo nllb33b sampling-mbr --n-resamples 1000 --out results/camera_ready/paired-bs.en-ar.json

# TASK B -- COMET paired bootstrap (sentence-level wmt22-comet-da, 1000 resamples, seed 12345)
python scripts/paired_bootstrap_comet.py --src Ar-En/dev.en --ref Ar-En/dev.ar --baseline predictions/mbr5pub.en-ar.dev.ar \
    --systems predictions/allam-ft.en-ar.dev.ar predictions/xalma-cpo.en-ar.dev.ar predictions/nllb33b.en-ar.dev.ar predictions/sampling-mbr.en-ar.dev.ar \
    --names allam-ft xalma-cpo nllb33b sampling-mbr --n-resamples 1000 --seed 12345 \
    --sentence-scores results/camera_ready/comet-sentence-scores.en-ar.json \
    --out results/camera_ready/paired-bs-comet.en-ar.json
```
