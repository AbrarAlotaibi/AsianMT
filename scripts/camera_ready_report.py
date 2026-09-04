"""Build camera_ready_experiments.md from results/camera_ready/*.json.

Reads whatever run_camera_ready.sh has produced so far; anything missing is
reported as "not computed" (never filled in by hand). Re-run after the GPU
job finishes to get the complete tables.

  python scripts/camera_ready_report.py [--out camera_ready_experiments.md]
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

POOL = ["nllb33b", "xalma-ft", "xalma-cpo", "allam-ft", "opus"]
COMPARE = ["allam-ft", "xalma-cpo", "nllb33b", "sampling-mbr"]
DIRS = {
    "ar-en": {"title": "Sub-Task 2A: Arabic -> English", "src": "Ar-En/dev.ar",
              "ref": "Ar-En/dev.en", "ext": "en"},
    "en-ar": {"title": "Sub-Task 1A: English -> Arabic", "src": "Ar-En/dev.en",
              "ref": "Ar-En/dev.ar", "ext": "ar"},
}
LABEL = {"nllb33b": "NLLB-3.3B FT", "xalma-ft": "X-ALMA-FT", "xalma-cpo": "X-ALMA-CPO",
         "allam-ft": "ALLaM-FT", "opus": "opus-mt", "sampling-mbr": "Sampling-MBR"}
RES = Path("results")
CR = RES / "camera_ready"
NA = "n/a"


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(x, nd=2):
    return NA if x is None else f"{x:.{nd}f}"


def sfmt(x, nd=2):
    return NA if x is None else f"{x:+.{nd}f}"


def task_a(d: str, lines: list[str], missing: list[str]) -> None:
    ext = DIRS[d]["ext"]
    full = jload(CR / f"loo-none.{d}.dev.json")
    recorded = jload(RES / f"mbr5pub.{d}.dev.json")
    rows = [("none (full 5-system pool)", full, None)]
    for s in POOL:
        r = jload(CR / f"loo-minus-{s}.{d}.dev.json")
        if r is None:
            missing.append(f"results/camera_ready/loo-minus-{s}.{d}.dev.json")
        rows.append((f"$-$\\,{LABEL[s]}", r, s))
    if full is None:
        missing.append(f"results/camera_ready/loo-none.{d}.dev.json")

    lines += [f"### {DIRS[d]['title']} (dev, n=1000)", ""]
    if recorded:
        lines += [f"Recorded full-pool score in the paper (`results/mbr5pub.{d}.dev.json`): "
                  f"BLEU {recorded['BLEU']:.2f} / chrF {recorded['chrF']:.2f} / "
                  f"TER {recorded['TER']:.2f} / COMET {recorded['COMET']:.4f}."]
        if full:
            same = all(abs(full[k] - recorded[k]) < 1e-9 for k in ("BLEU", "chrF", "TER", "COMET"))
            lines.append("Re-generated full pool " + ("**matches** the recorded row exactly."
                         if same else "**differs** from the recorded row -- check before use."))
        lines.append("")
    lines += ["```latex", "\\begin{tabular}{lrrrrrr}", "\\toprule",
              "Pool removed & BLEU & chrF & TER & COMET & $\\Delta$BLEU & $\\Delta$COMET \\\\",
              "\\midrule"]
    for name, r, _ in rows:
        if r is None:
            lines.append(f"{name} & {NA} & {NA} & {NA} & {NA} & {NA} & {NA} \\\\")
            continue
        db = dc = None
        if full is not None:
            db, dc = r["BLEU"] - full["BLEU"], r["COMET"] - full["COMET"]
        lines.append(f"{name} & {r['BLEU']:.2f} & {r['chrF']:.2f} & {r['TER']:.2f} & "
                     f"{r['COMET']:.4f} & {sfmt(db)} & {sfmt(dc, 4)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "```", ""]
    if full and all(r for _, r, _ in rows):
        worst = min((r for _, r, s in rows if s), key=lambda r: r["COMET"])
        best = max((r for _, r, s in rows if s), key=lambda r: r["COMET"])
        drop_worst = [s for _, r, s in rows if s and r is worst][0]
        drop_best = [s for _, r, s in rows if s and r is best][0]
        lines += [f"Largest COMET loss when removing **{LABEL[drop_worst]}** "
                  f"({worst['COMET'] - full['COMET']:+.4f}); smallest when removing "
                  f"**{LABEL[drop_best]}** ({best['COMET'] - full['COMET']:+.4f}).", ""]
    lines += ["Hypotheses: `predictions/mbr5pub." + f"{d}.dev.{ext}` (full pool), "
              f"`predictions/camera_ready/mbr-minus-<sys>.{d}.dev.{ext}` (leave-one-out).", ""]


def task_b(d: str, lines: list[str], missing: list[str], flagged: list[str]) -> None:
    sb = jload(CR / f"paired-bs.{d}.json")
    cm = jload(CR / f"paired-bs-comet.{d}.json")
    if sb is None:
        missing.append(f"results/camera_ready/paired-bs.{d}.json")
    if cm is None:
        missing.append(f"results/camera_ready/paired-bs-comet.{d}.json")
    lines += [f"### {DIRS[d]['title']} (dev, n=1000)", "",
              "Baseline = full-pool MBR primary. delta = system $-$ primary "
              "(so a negative delta on BLEU/chrF/COMET, or positive on TER, "
              "means the primary is better).", "",
              "| Comparison | Metric | primary | system | delta | p | sig. |",
              "|---|---|---:|---:|---:|---:|:---:|"]
    sb_sys = {s["name"]: s for s in sb["systems"]} if sb else {}
    cm_sys = {s["name"]: s for s in cm["systems"]} if cm else {}
    for name in COMPARE:
        for m in ("BLEU", "chrF", "TER"):
            r = sb_sys.get(name, {}).get(m)
            if r is None:
                lines.append(f"| primary vs {LABEL[name]} | {m} | {NA} | {NA} | {NA} | {NA} | |")
                continue
            sig = "*" if r["p"] < 0.05 else "**n.s.**"
            if r["p"] >= 0.05:
                flagged.append(f"{d}: primary vs {LABEL[name]} on {m} (delta {r['delta']:+.2f}, p = {r['p']:.4f})")
            lines.append(f"| primary vs {LABEL[name]} | {m} | {sb['baseline'][m]:.2f} | "
                         f"{r['score']:.2f} | {r['delta']:+.2f} | {r['p']:.4f} | {sig} |")
        r = cm_sys.get(name)
        if r is None:
            lines.append(f"| primary vs {LABEL[name]} | COMET | {NA} | {NA} | {NA} | {NA} | |")
            continue
        sig = "*" if r["p"] < 0.05 else "**n.s.**"
        if r["p"] >= 0.05:
            flagged.append(f"{d}: primary vs {LABEL[name]} on COMET (delta {r['delta']:+.4f}, p = {r['p']:.4f})")
        lines.append(f"| primary vs {LABEL[name]} | COMET | {cm['baseline']['COMET']:.4f} | "
                     f"{r['COMET']:.4f} | {r['delta']:+.4f} | {r['p']:.4f} | {sig} |")
    lines.append("")

    lines += ["```latex", "\\begin{tabular}{llrr}", "\\toprule",
              "Comparison & Metric & $\\Delta$ & $p$ \\\\", "\\midrule"]
    for name in COMPARE:
        cells = []
        for m in ("BLEU", "chrF", "TER"):
            r = sb_sys.get(name, {}).get(m)
            cells.append((m, None, None) if r is None else (m, r["delta"], r["p"]))
        r = cm_sys.get(name)
        cells.append(("COMET", None, None) if r is None else ("COMET", r["delta"], r["p"]))
        for i, (m, delta, p) in enumerate(cells):
            comp = f"primary vs.\\ {LABEL[name]}" if i == 0 else ""
            if delta is None:
                lines.append(f"{comp} & {m} & {NA} & {NA} \\\\")
            else:
                nd = 4 if m == "COMET" else 2
                ptxt = f"{p:.3f}" if p >= 0.001 else "$<$0.001"
                mark = "" if p < 0.05 else "$^{\\dagger}$"
                lines.append(f"{comp} & {m} & {delta:+.{nd}f} & {ptxt}{mark} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "```", "",
              "$^{\\dagger}$ = not significant at $p<0.05$.", ""]
    if sb:
        lines += ["sacreBLEU signatures (paired bootstrap):", ""]
        lines += [f"- {m}: `{s}`" for m, s in sb["signatures"].items()]
        lines.append("")
    if cm:
        lines += [f"COMET bootstrap: `{cm['model']}`, {cm['n_resamples']} resamples, "
                  f"seed {cm['seed']}, sentence scores cached in "
                  f"`results/camera_ready/comet-sentence-scores.{d}.json`.", ""]


def commands(d: str) -> list[str]:
    c = DIRS[d]
    ext, src, ref = c["ext"], c["src"], c["ref"]
    pool = [f"predictions/{s}.{d}.dev.{ext}" for s in POOL]
    others = [f"predictions/{s}.{d}.dev.{ext}" for s in COMPARE]
    full = f"predictions/mbr5pub.{d}.dev.{ext}"
    out = [f"#### {d}", "", "```bash", "# TASK A -- full pool (reference row)",
           f"python scripts/mbr_decode.py --src {src} \\",
           "    --candidates " + " ".join(pool) + " \\",
           f"    --output {full}",
           f"python scripts/evaluate.py --hyp {full} --ref {ref} --src {src} --comet \\",
           f"    > results/camera_ready/loo-none.{d}.dev.json", ""]
    for s in POOL:
        p4 = [x for x in pool if f"/{s}." not in x]
        o = f"predictions/camera_ready/mbr-minus-{s}.{d}.dev.{ext}"
        out += [f"# TASK A -- pool without {s}",
                f"python scripts/mbr_decode.py --src {src} \\",
                "    --candidates " + " ".join(p4) + " \\",
                f"    --output {o}",
                f"python scripts/evaluate.py --hyp {o} --ref {ref} --src {src} --comet \\",
                f"    > results/camera_ready/loo-minus-{s}.{d}.dev.json", ""]
    out += ["# TASK B -- BLEU/chrF/TER paired bootstrap (1000 resamples, default seed 12345)",
            f"sacrebleu {ref} -i {full} \\",
            "    " + " \\\n    ".join(others) + " \\",
            f"    -m bleu chrf ter --paired-bs --paired-bs-n 1000 -f text > results/camera_ready/paired-bs.{d}.txt",
            "# same test through the sacreBLEU API, full-precision JSON:",
            f"python scripts/paired_bootstrap_sacrebleu.py --ref {ref} --baseline {full} \\",
            "    --systems " + " ".join(others) + " \\",
            "    --names " + " ".join(COMPARE) + f" --n-resamples 1000 --out results/camera_ready/paired-bs.{d}.json",
            "", "# TASK B -- COMET paired bootstrap (sentence-level wmt22-comet-da, 1000 resamples, seed 12345)",
            f"python scripts/paired_bootstrap_comet.py --src {src} --ref {ref} --baseline {full} \\",
            "    --systems " + " ".join(others) + " \\",
            "    --names " + " ".join(COMPARE) + " --n-resamples 1000 --seed 12345 \\",
            f"    --sentence-scores results/camera_ready/comet-sentence-scores.{d}.json \\",
            f"    --out results/camera_ready/paired-bs-comet.{d}.json",
            "```", ""]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("camera_ready_experiments.md"))
    args = ap.parse_args()

    missing_results: list[str] = []
    flagged: list[str] = []
    L: list[str] = []

    inputs = []
    for d, c in DIRS.items():
        inputs += [c["src"], c["ref"]]
        inputs += [f"predictions/{s}.{d}.dev.{c['ext']}" for s in POOL + ["sampling-mbr"]]
        inputs.append(f"predictions/mbr5pub.{d}.dev.{c['ext']}")
    missing_inputs = [f for f in inputs if not Path(f).exists()]

    L += ["# Camera-Ready Experiments: MBR Pool Ablation and Significance Tests", "",
          f"Generated {date.today().isoformat()} by `scripts/camera_ready_report.py` "
          "from `results/camera_ready/`. Official DEV set, n=1000 per direction. "
          "Every number here is read from a result file written by the commands in "
          "Section 4; nothing is filled in by hand.", ""]

    L += ["## 0. Status", ""]
    if missing_inputs:
        L += ["**Missing input files** (the runs cannot start until these exist):", ""]
        L += [f"- `{f}`" for f in missing_inputs]
        L += ["", "`predictions/mbr5pub.*.dev.*` are the full-pool MBR dev hypotheses: "
              "they were produced by `run_mbr5_dev.sh` on the GPU server but never "
              "committed (`push_results.sh` only stages `results/`), and the server "
              "session is gone. `run_camera_ready.sh` regenerates them with "
              "`scripts/mbr_decode.py` (deterministic given the model), then checks "
              "the scores against the recorded `results/mbr5pub.*.dev.json`.", ""]
    else:
        L += ["All input files present.", ""]
    L += ["To produce every table below: `CUDA_VISIBLE_DEVICES=0 bash run_camera_ready.sh` "
          "(GPU; COMET-22 MBR over 12 pools + COMET scoring of 12 outputs + 8 sentence-level "
          "COMET passes, roughly 30-45 min on one RTX 3090), then re-run "
          "`python scripts/camera_ready_report.py`.", ""]

    L += ["## 1. Setup", "",
          "- Pool (both directions): NLLB-3.3B FT, X-ALMA-FT, X-ALMA-CPO, ALLaM-FT, opus-mt "
          "(`nllb33b`, `xalma-ft`, `xalma-cpo`, `allam-ft`, `opus`).",
          "- MBR: `scripts/mbr_decode.py`, utility `Unbabel/wmt22-comet-da`, no QE pruning, "
          "per-sentence dedup of identical candidates, expected utility over the pool "
          "(self-pair included).",
          "- Metrics: sacreBLEU 2.6.0 defaults via `scripts/evaluate.py`; "
          "signature `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`; "
          "COMET `Unbabel/wmt22-comet-da`.",
          "- References: raw `Ar-En/dev.{ar,en}` (the recorded per-system dev scores "
          "reproduce exactly against these files with sacreBLEU 2.6.0).",
          "- Paired bootstrap: 1000 resamples with replacement, sacreBLEU default seed "
          "12345, p-value as in sacreBLEU (`(#{shifted |delta_b| > |delta|} + 1) / (N + 1)`); "
          "the COMET test applies the identical procedure to sentence-level "
          "wmt22-comet-da scores (`scripts/paired_bootstrap_comet.py`).", ""]

    L += ["## 2. Task A -- Leave-one-out ablation of the MBR pool", ""]
    for d in DIRS:
        task_a(d, L, missing_results)

    L += ["## 3. Task B -- Paired bootstrap significance vs. the primary", ""]
    for d in DIRS:
        task_b(d, L, missing_results, flagged)
    L += ["### Comparisons with p >= 0.05", ""]
    if flagged:
        L += [f"- {f}" for f in flagged]
    elif missing_results:
        L += ["(not yet computed)"]
    else:
        L += ["None -- every comparison is significant at p < 0.05."]
    L.append("")

    L += ["## 4. Exact commands", "",
          "All paths are relative to the repository root. `run_camera_ready.sh` runs "
          "exactly these for both directions and then this report script.", ""]
    for d in DIRS:
        L += commands(d)

    if missing_results:
        L += ["## 5. Result files not yet produced", ""]
        L += [f"- `{f}`" for f in sorted(set(missing_results))]
        L.append("")

    args.out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out}  (missing inputs: {len(missing_inputs)}, "
          f"missing result files: {len(set(missing_results))}, flagged p>=0.05: {len(flagged)})")


if __name__ == "__main__":
    main()
