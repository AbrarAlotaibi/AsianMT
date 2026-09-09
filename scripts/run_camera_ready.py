"""Windows-friendly driver for the camera-ready experiments.

Runs exactly the commands of ``run_camera_ready.sh`` (same inputs, same
outputs, same metric settings) through native subprocess calls instead of a
Git-Bash pipeline. Written for a Windows box (i9-14900KF, RTX 4080 SUPER)
where the CPU-bound sacreBLEU steps randomly died with access violations on
the P-cores (Raptor Lake instability) and detached MSYS bash sometimes failed
to spawn steps. Every step is deterministic and writes its output only on
success, so a crashed step is simply re-run (up to --retries times), and the
driver pins itself (and its children) to the E-cores on Windows.

  python scripts/run_camera_ready.py            # both directions + report
  python scripts/run_camera_ready.py --dirs ar-en

Steps already completed (valid result JSON / hypothesis file present) are
skipped, so the driver can be re-launched after an interruption.
Log: results/camera_ready/run.log
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "camera_ready"
LOO = ROOT / "predictions" / "camera_ready"
SYS = ["nllb33b", "xalma-ft", "xalma-cpo", "allam-ft", "opus"]
COMPARE = ["allam-ft", "xalma-cpo", "nllb33b", "sampling-mbr"]
DIRS = {
    "ar-en": {"src": "Ar-En/dev.ar", "ref": "Ar-En/dev.en", "ext": "en"},
    "en-ar": {"src": "Ar-En/dev.en", "ref": "Ar-En/dev.ar", "ext": "ar"},
}
N_BS = 1000
CRASH_CODES = {139, 133, 134, -11, -6, 3221225477, -1073741819, 3221226505, -1073740791}

LOG = None


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG:
        LOG.write(line + "\n")
        LOG.flush()


def run(cmd: list[str], stdout_file: Path | None, retries: int) -> None:
    """Run cmd from ROOT; retry on crash exit codes. stdout -> file if given."""
    for attempt in range(1, retries + 1):
        log("$ " + " ".join(cmd) + (f"  > {stdout_file.relative_to(ROOT)}" if stdout_file else ""))
        t0 = time.time()
        if stdout_file:
            tmp = stdout_file.with_suffix(stdout_file.suffix + ".part")
            with open(tmp, "w", encoding="utf-8") as fo:
                p = subprocess.run(cmd, cwd=ROOT, stdout=fo, stderr=LOG, text=True)
            if p.returncode == 0:
                tmp.replace(stdout_file)
            else:
                tmp.unlink(missing_ok=True)
        else:
            p = subprocess.run(cmd, cwd=ROOT, stdout=LOG, stderr=subprocess.STDOUT, text=True)
        log(f"  -> rc={p.returncode} in {time.time() - t0:.0f}s")
        if p.returncode == 0:
            return
        if p.returncode in CRASH_CODES or p.returncode < 0:
            log(f"  crash on attempt {attempt}/{retries}; retrying")
            time.sleep(3)
            continue
        sys.exit(f"ERROR: step failed with rc={p.returncode}: {' '.join(cmd)}")
    sys.exit(f"ERROR: step crashed {retries} times: {' '.join(cmd)}")


def valid_json(p: Path, key: str = "COMET") -> bool:
    try:
        return key in json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False


def run_dir(d: str, py: str, retries: int) -> None:
    c = DIRS[d]
    src, ref, ext = c["src"], c["ref"], c["ext"]
    log(f"################ {d} ################")
    missing = [f for f in [src, ref] + [f"predictions/{s}.{d}.dev.{ext}" for s in SYS + ["sampling-mbr"]]
               if not (ROOT / f).exists()]
    if missing:
        sys.exit("ERROR: missing inputs: " + ", ".join(missing))

    def mbr(pool: list[str], out: Path) -> None:
        if out.exists() and sum(1 for _ in open(out, encoding="utf-8")) == 1000:
            log(f"  skip MBR (exists): {out.relative_to(ROOT)}")
            return
        run([py, "scripts/mbr_decode.py", "--src", src, "--candidates", *pool, "--output", str(out)],
            None, retries)

    def score(hyp: Path, res: Path) -> None:
        if valid_json(res):
            log(f"  skip score (exists): {res.relative_to(ROOT)}")
            return
        run([py, "scripts/evaluate.py", "--hyp", str(hyp), "--ref", ref, "--src", src, "--comet"],
            res, retries)
        log("  " + json.dumps({k: v for k, v in json.loads(res.read_text(encoding='utf-8')).items()
                               if k in ("BLEU", "chrF", "TER", "COMET")}))

    # TASK A: full pool + leave-one-out pools
    full = ROOT / f"predictions/mbr5pub.{d}.dev.{ext}"
    cands = [f"predictions/{s}.{d}.dev.{ext}" for s in SYS]
    mbr(cands, full)
    score(full, OUT / f"loo-none.{d}.dev.json")
    for drop in SYS:
        pool = [f"predictions/{s}.{d}.dev.{ext}" for s in SYS if s != drop]
        out = LOO / f"mbr-minus-{drop}.{d}.dev.{ext}"
        mbr(pool, out)
        score(out, OUT / f"loo-minus-{drop}.{d}.dev.json")

    # TASK B: paired bootstrap, primary vs 4 systems
    others = [f"predictions/{s}.{d}.dev.{ext}" for s in COMPARE]
    txt = OUT / f"paired-bs.{d}.txt"
    if not (txt.exists() and txt.stat().st_size > 0):
        run([py, "-m", "sacrebleu", ref, "-i", str(full), *others, "-m", "bleu", "chrf", "ter",
             "--paired-bs", "--paired-bs-n", str(N_BS), "-f", "text"], txt, retries)
    else:
        log(f"  skip (exists): {txt.relative_to(ROOT)}")
    js = OUT / f"paired-bs.{d}.json"
    if not valid_json(js, "systems"):
        run([py, "scripts/paired_bootstrap_sacrebleu.py", "--ref", ref, "--baseline", str(full),
             "--systems", *others, "--names", *COMPARE, "--n-resamples", str(N_BS), "--out", str(js)],
            None, retries)
    else:
        log(f"  skip (exists): {js.relative_to(ROOT)}")
    cj = OUT / f"paired-bs-comet.{d}.json"
    if not valid_json(cj, "systems"):
        run([py, "scripts/paired_bootstrap_comet.py", "--src", src, "--ref", ref, "--baseline", str(full),
             "--systems", *others, "--names", *COMPARE, "--n-resamples", str(N_BS), "--seed", "12345",
             "--sentence-scores", str(OUT / f"comet-sentence-scores.{d}.json"), "--out", str(cj)],
            None, retries)
    else:
        log(f"  skip (exists): {cj.relative_to(ROOT)}")


def pin_to_ecores(first_cpu: int) -> str:
    """Windows only: restrict this process (and, by inheritance, every child)
    to logical CPUs >= first_cpu. On an i9-14900KF those are the E-cores;
    the P-cores on this box randomly raise access violations in
    allocation-heavy pure-Python code (Raptor Lake Vmin-shift instability,
    BIOS 1.60 / 2023 microcode), which is where all the crashes came from."""
    if os.name != "nt":
        return "not windows"
    import ctypes
    n = os.cpu_count() or 0
    if n <= first_cpu:
        return f"cpu_count={n} <= {first_cpu}; not pinning"
    mask = sum(1 << i for i in range(first_cpu, n))
    ok = ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(),
                                                       ctypes.c_size_t(mask))
    return f"pinned to logical CPUs {first_cpu}-{n - 1}" if ok else "SetProcessAffinityMask failed"


def main() -> None:
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=list(DIRS), choices=list(DIRS))
    ap.add_argument("--retries", type=int, default=25)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--ecores-from", type=int, default=16,
                    help="Windows: pin to logical CPUs >= this (E-cores on 14900K). -1 = no pinning.")
    args = ap.parse_args()
    pin_msg = pin_to_ecores(args.ecores_from) if args.ecores_from >= 0 else "no pinning"

    OUT.mkdir(parents=True, exist_ok=True)
    LOO.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    LOG = open(OUT / "run.log", "a", encoding="utf-8")
    log(f"=== run_camera_ready.py {datetime.now(timezone.utc).strftime('%FT%TZ')} "
        f"python={args.python} GPU={os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')} affinity: {pin_msg} ===")
    for d in args.dirs:
        run_dir(d, args.python, args.retries)
    log("building camera_ready_experiments.md")
    run([args.python, "scripts/camera_ready_report.py"], None, args.retries)
    log("done.")
    (OUT / "DONE").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")


if __name__ == "__main__":
    main()
