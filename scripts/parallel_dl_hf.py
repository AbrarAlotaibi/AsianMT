#!/usr/bin/env python
"""Generic parallel ranged downloader for a single (possibly gated) HF file.

Same rationale as parallel_dl_comet.py: per-connection egress throttle + xet
hangs, so fetch the file in many concurrent byte-ranges, each re-hitting the
/resolve/ URL so HF re-signs the 1h CDN signature. Adds an Authorization
header for gated repos. Resumable + sha256-verified, then wired into the HF
cache (blob + snapshot symlink) so download_model()/from_pretrained() are no-ops.

Usage:
  python parallel_dl_hf.py <repo> <path_in_repo> <sha256> <size> <commit>
  Token read from env HF_TOKEN (optional for public files).
"""
import concurrent.futures as cf
import hashlib
import os
import sys
import threading
import time

import requests

REPO, RELPATH, SHA256, SIZE, COMMIT = sys.argv[1:6]
SIZE = int(SIZE)
TOKEN = os.environ.get("HF_TOKEN", "")
URL = f"https://huggingface.co/{REPO}/resolve/main/{RELPATH}"
HEADERS_AUTH = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}

repo_dir = "models--" + REPO.replace("/", "--")
CACHE = os.path.expanduser(f"~/.cache/huggingface/hub/{repo_dir}")
BLOB = os.path.join(CACHE, "blobs", SHA256)
PART = BLOB + ".part"
DONE_MARK = PART + ".done"
LINK = os.path.join(CACHE, "snapshots", COMMIT, RELPATH)

CHUNK = 32 * 1024 * 1024
WORKERS = 16


def completed_chunks():
    if os.path.exists(DONE_MARK):
        with open(DONE_MARK) as f:
            return set(int(x) for x in f.read().split() if x.strip())
    return set()


def mark_done(idx, lock):
    with lock:
        with open(DONE_MARK, "a") as f:
            f.write(f"{idx}\n")


def fetch(idx, start, end, lock):
    headers = dict(HEADERS_AUTH)
    headers["Range"] = f"bytes={start}-{end}"
    last = None
    for attempt in range(8):
        try:
            r = requests.get(URL, headers=headers, stream=True, timeout=60,
                             allow_redirects=True)
            if r.status_code not in (200, 206):
                raise RuntimeError(f"status {r.status_code}")
            with open(PART, "r+b") as f:
                f.seek(start)
                for block in r.iter_content(1024 * 256):
                    if block:
                        f.write(block)
            mark_done(idx, lock)
            return idx
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"chunk {idx} failed: {last}")


def main():
    os.makedirs(os.path.dirname(BLOB), exist_ok=True)
    if not os.path.exists(PART) or os.path.getsize(PART) != SIZE:
        if os.path.exists(PART):
            os.remove(PART)
        if os.path.exists(DONE_MARK):
            os.remove(DONE_MARK)
        with open(PART, "wb") as f:
            f.truncate(SIZE)

    ranges, idx, pos = [], 0, 0
    while pos < SIZE:
        end = min(pos + CHUNK - 1, SIZE - 1)
        ranges.append((idx, pos, end))
        pos = end + 1
        idx += 1
    total = len(ranges)
    done = completed_chunks()
    todo = [r for r in ranges if r[0] not in done]
    print(f"total chunks={total} already_done={len(done)} todo={len(todo)}",
          flush=True)

    lock = threading.Lock()
    t0 = time.time()
    completed = len(done)
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(fetch, i, s, e, lock) for (i, s, e) in todo]
        for fut in cf.as_completed(futs):
            fut.result()
            completed += 1
            el = time.time() - t0
            got = (completed - len(done)) * CHUNK
            rate = got / el / 1048576 if el > 0 else 0
            print(f"[{completed}/{total} {100*completed/total:5.1f}%] "
                  f"~{rate:5.2f} MB/s elapsed={el/60:.1f}m", flush=True)

    print("verifying sha256 ...", flush=True)
    h = hashlib.sha256()
    with open(PART, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != SHA256:
        print(f"SHA MISMATCH got={h.hexdigest()} want={SHA256}", flush=True)
        sys.exit(1)
    print("sha256 OK", flush=True)

    os.replace(PART, BLOB)
    if os.path.exists(DONE_MARK):
        os.remove(DONE_MARK)
    os.makedirs(os.path.dirname(LINK), exist_ok=True)
    rel = os.path.relpath(BLOB, os.path.dirname(LINK))
    if os.path.islink(LINK) or os.path.exists(LINK):
        os.remove(LINK)
    os.symlink(rel, LINK)
    # point refs/main at the commit too (snapshot_download expects it)
    refs = os.path.join(CACHE, "refs")
    os.makedirs(refs, exist_ok=True)
    with open(os.path.join(refs, "main"), "w") as f:
        f.write(COMMIT)
    print(f"BLOB ready: {BLOB}", flush=True)
    print(f"SYMLINK: {LINK} -> {rel}", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
