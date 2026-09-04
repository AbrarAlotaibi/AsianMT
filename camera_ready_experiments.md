# Camera-Ready Experiments: MBR Pool Ablation and Significance Tests

Generated 2026-09-04 by `scripts/camera_ready_report.py` from `results/camera_ready/`. Official DEV set, n=1000 per direction. Every number here is read from a result file written by the commands in Section 4; nothing is filled in by hand.

## 0. Status

**Missing input files** (the runs cannot start until these exist):

- `predictions/mbr5pub.ar-en.dev.en`
- `predictions/mbr5pub.en-ar.dev.ar`

`predictions/mbr5pub.*.dev.*` are the full-pool MBR dev hypotheses: they were produced by `run_mbr5_dev.sh` on the GPU server but never committed (`push_results.sh` only stages `results/`), and the server session is gone. `run_camera_ready.sh` regenerates them with `scripts/mbr_decode.py` (deterministic given the model), then checks the scores against the recorded `results/mbr5pub.*.dev.json`.

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

```latex
\begin{tabular}{lrrrrrr}
\toprule
Pool removed & BLEU & chrF & TER & COMET & $\Delta$BLEU & $\Delta$COMET \\
\midrule
none (full 5-system pool) & n/a & n/a & n/a & n/a & n/a & n/a \\
$-$\,NLLB-3.3B FT & n/a & n/a & n/a & n/a & n/a & n/a \\
$-$\,X-ALMA-FT & n/a & n/a & n/a & n/a & n/a & n/a \\
$-$\,X-ALMA-CPO & n/a & n/a & n/a & n/a & n/a & n/a \\
$-$\,ALLaM-FT & n/a & n/a & n/a & n/a & n/a & n/a \\
$-$\,opus-mt & n/a & n/a & n/a & n/a & n/a & n/a \\
\bottomrule
\end{tabular}
```

Hypotheses: `predictions/mbr5pub.ar-en.dev.en` (full pool), `predictions/camera_ready/mbr-minus-<sys>.ar-en.dev.en` (leave-one-out).

### Sub-Task 1A: English -> Arabic (dev, n=1000)

Recorded full-pool score in the paper (`results/mbr5pub.en-ar.dev.json`): BLEU 21.48 / chrF 54.90 / TER 68.97 / COMET 0.8865.

```latex
\begin{tabular}{lrrrrrr}
\toprule
Pool removed & BLEU & chrF & TER & COMET & $\Delta$BLEU & $\Delta$COMET \\
\midrule
none (full 5-system pool) & n/a & n/a & n/a & n/a & n/a & n/a \\
$-$\,NLLB-3.3B FT & n/a & n/a & n/a & n/a & n/a & n/a \\
$-$\,X-ALMA-FT & n/a & n/a & n/a & n/a & n/a & n/a \\
$-$\,X-ALMA-CPO & n/a & n/a & n/a & n/a & n/a & n/a \\
$-$\,ALLaM-FT & n/a & n/a & n/a & n/a & n/a & n/a \\
$-$\,opus-mt & n/a & n/a & n/a & n/a & n/a & n/a \\
\bottomrule
\end{tabular}
```

Hypotheses: `predictions/mbr5pub.en-ar.dev.ar` (full pool), `predictions/camera_ready/mbr-minus-<sys>.en-ar.dev.ar` (leave-one-out).

## 3. Task B -- Paired bootstrap significance vs. the primary

### Sub-Task 2A: Arabic -> English (dev, n=1000)

Baseline = full-pool MBR primary. delta = system $-$ primary (so a negative delta on BLEU/chrF/COMET, or positive on TER, means the primary is better).

| Comparison | Metric | primary | system | delta | p | sig. |
|---|---|---:|---:|---:|---:|:---:|
| primary vs ALLaM-FT | BLEU | n/a | n/a | n/a | n/a | |
| primary vs ALLaM-FT | chrF | n/a | n/a | n/a | n/a | |
| primary vs ALLaM-FT | TER | n/a | n/a | n/a | n/a | |
| primary vs ALLaM-FT | COMET | n/a | n/a | n/a | n/a | |
| primary vs X-ALMA-CPO | BLEU | n/a | n/a | n/a | n/a | |
| primary vs X-ALMA-CPO | chrF | n/a | n/a | n/a | n/a | |
| primary vs X-ALMA-CPO | TER | n/a | n/a | n/a | n/a | |
| primary vs X-ALMA-CPO | COMET | n/a | n/a | n/a | n/a | |
| primary vs NLLB-3.3B FT | BLEU | n/a | n/a | n/a | n/a | |
| primary vs NLLB-3.3B FT | chrF | n/a | n/a | n/a | n/a | |
| primary vs NLLB-3.3B FT | TER | n/a | n/a | n/a | n/a | |
| primary vs NLLB-3.3B FT | COMET | n/a | n/a | n/a | n/a | |
| primary vs Sampling-MBR | BLEU | n/a | n/a | n/a | n/a | |
| primary vs Sampling-MBR | chrF | n/a | n/a | n/a | n/a | |
| primary vs Sampling-MBR | TER | n/a | n/a | n/a | n/a | |
| primary vs Sampling-MBR | COMET | n/a | n/a | n/a | n/a | |

```latex
\begin{tabular}{llrr}
\toprule
Comparison & Metric & $\Delta$ & $p$ \\
\midrule
primary vs.\ ALLaM-FT & BLEU & n/a & n/a \\
 & chrF & n/a & n/a \\
 & TER & n/a & n/a \\
 & COMET & n/a & n/a \\
\midrule
primary vs.\ X-ALMA-CPO & BLEU & n/a & n/a \\
 & chrF & n/a & n/a \\
 & TER & n/a & n/a \\
 & COMET & n/a & n/a \\
\midrule
primary vs.\ NLLB-3.3B FT & BLEU & n/a & n/a \\
 & chrF & n/a & n/a \\
 & TER & n/a & n/a \\
 & COMET & n/a & n/a \\
\midrule
primary vs.\ Sampling-MBR & BLEU & n/a & n/a \\
 & chrF & n/a & n/a \\
 & TER & n/a & n/a \\
 & COMET & n/a & n/a \\
\bottomrule
\end{tabular}
```

$^{\dagger}$ = not significant at $p<0.05$.

### Sub-Task 1A: English -> Arabic (dev, n=1000)

Baseline = full-pool MBR primary. delta = system $-$ primary (so a negative delta on BLEU/chrF/COMET, or positive on TER, means the primary is better).

| Comparison | Metric | primary | system | delta | p | sig. |
|---|---|---:|---:|---:|---:|:---:|
| primary vs ALLaM-FT | BLEU | n/a | n/a | n/a | n/a | |
| primary vs ALLaM-FT | chrF | n/a | n/a | n/a | n/a | |
| primary vs ALLaM-FT | TER | n/a | n/a | n/a | n/a | |
| primary vs ALLaM-FT | COMET | n/a | n/a | n/a | n/a | |
| primary vs X-ALMA-CPO | BLEU | n/a | n/a | n/a | n/a | |
| primary vs X-ALMA-CPO | chrF | n/a | n/a | n/a | n/a | |
| primary vs X-ALMA-CPO | TER | n/a | n/a | n/a | n/a | |
| primary vs X-ALMA-CPO | COMET | n/a | n/a | n/a | n/a | |
| primary vs NLLB-3.3B FT | BLEU | n/a | n/a | n/a | n/a | |
| primary vs NLLB-3.3B FT | chrF | n/a | n/a | n/a | n/a | |
| primary vs NLLB-3.3B FT | TER | n/a | n/a | n/a | n/a | |
| primary vs NLLB-3.3B FT | COMET | n/a | n/a | n/a | n/a | |
| primary vs Sampling-MBR | BLEU | n/a | n/a | n/a | n/a | |
| primary vs Sampling-MBR | chrF | n/a | n/a | n/a | n/a | |
| primary vs Sampling-MBR | TER | n/a | n/a | n/a | n/a | |
| primary vs Sampling-MBR | COMET | n/a | n/a | n/a | n/a | |

```latex
\begin{tabular}{llrr}
\toprule
Comparison & Metric & $\Delta$ & $p$ \\
\midrule
primary vs.\ ALLaM-FT & BLEU & n/a & n/a \\
 & chrF & n/a & n/a \\
 & TER & n/a & n/a \\
 & COMET & n/a & n/a \\
\midrule
primary vs.\ X-ALMA-CPO & BLEU & n/a & n/a \\
 & chrF & n/a & n/a \\
 & TER & n/a & n/a \\
 & COMET & n/a & n/a \\
\midrule
primary vs.\ NLLB-3.3B FT & BLEU & n/a & n/a \\
 & chrF & n/a & n/a \\
 & TER & n/a & n/a \\
 & COMET & n/a & n/a \\
\midrule
primary vs.\ Sampling-MBR & BLEU & n/a & n/a \\
 & chrF & n/a & n/a \\
 & TER & n/a & n/a \\
 & COMET & n/a & n/a \\
\bottomrule
\end{tabular}
```

$^{\dagger}$ = not significant at $p<0.05$.

### Comparisons with p >= 0.05

(not yet computed)

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

## 5. Result files not yet produced

- `results/camera_ready/loo-minus-allam-ft.ar-en.dev.json`
- `results/camera_ready/loo-minus-allam-ft.en-ar.dev.json`
- `results/camera_ready/loo-minus-nllb33b.ar-en.dev.json`
- `results/camera_ready/loo-minus-nllb33b.en-ar.dev.json`
- `results/camera_ready/loo-minus-opus.ar-en.dev.json`
- `results/camera_ready/loo-minus-opus.en-ar.dev.json`
- `results/camera_ready/loo-minus-xalma-cpo.ar-en.dev.json`
- `results/camera_ready/loo-minus-xalma-cpo.en-ar.dev.json`
- `results/camera_ready/loo-minus-xalma-ft.ar-en.dev.json`
- `results/camera_ready/loo-minus-xalma-ft.en-ar.dev.json`
- `results/camera_ready/loo-none.ar-en.dev.json`
- `results/camera_ready/loo-none.en-ar.dev.json`
- `results/camera_ready/paired-bs-comet.ar-en.json`
- `results/camera_ready/paired-bs-comet.en-ar.json`
- `results/camera_ready/paired-bs.ar-en.json`
- `results/camera_ready/paired-bs.en-ar.json`
