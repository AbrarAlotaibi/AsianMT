#!/usr/bin/env python
"""Parallel ranged downloader for the wmt22-comet-da checkpoint.

The host's egress is throttled per-connection (~30 KB/s) but allows many
connections in parallel, and HF's xet downloader hangs. So we fetch the file
in many byte-ranges concurrently, each request re-hitting the /resolve/ URL so
HuggingFace re-signs (the CDN signature expires in 1h). Resumable + verified.
"""
import concurrent.futures as cf
import hashlib
import os
import sys
import time

import requests

URL = "https://huggingface.co/Unbabel/wmt22-comet-da/resolve/main/checkpoints/model.ckpt"
SHA256 = "e213091cde220f97b89f8bdfa750c458cfea741ad62affb455b59900210ff2af"
SIZE = 2323649073
CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Unbabel--wmt22-comet-da"
)
BLOB = os.path.join(CACHE, "blobs", SHA256)
PART = BLOB + ".part"
SNAP = os.path.join(CACHE, "snapshots", "2760a223ac957f30acfb18c8aa649b01cf1d75f2")
LINK = os.path.join(SNAP, "checkpoints", "model.ckpt")

CHUNK = 32 * 1024 * 1024          # 32 MB per range job
WORKERS = 16
DONE_MARK = PART + ".done"        # records completed chunk indices


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
    headers = {"Range": f"bytes={start}-{end}"}
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
            return idx, end - start + 1
        except Exception as e:  # noqa: BLE001
            time.sleep(min(2 ** attempt, 30))
            last = e
    raise RuntimeError(f"chunk {idx} failed: {last}")


def main():
    # preallocate the part file to full size if needed
    if not os.path.exists(PART) or os.path.getsize(PART) != SIZE:
        if not os.path.exists(PART):
            with open(PART, "wb") as f:
                f.truncate(SIZE)
        elif os.path.getsize(PART) != SIZE:
            # wrong size -> restart clean
            os.remove(PART)
            if os.path.exists(DONE_MARK):
                os.remove(DONE_MARK)
            with open(PART, "wb") as f:
                f.truncate(SIZE)

    ranges = []
    idx = 0
    pos = 0
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

    import threading
    lock = threading.Lock()
    start_t = time.time()
    completed = len(done)
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, i, s, e, lock): i for (i, s, e) in todo}
        for fut in cf.as_completed(futs):
            fut.result()
            completed += 1
            elapsed = time.time() - start_t
            got = (completed - len(done)) * CHUNK
            rate = got / elapsed / 1048576 if elapsed > 0 else 0
            pct = 100 * completed / total
            print(f"[{completed}/{total} {pct:5.1f}%] ~{rate:5.2f} MB/s "
                  f"elapsed={elapsed/60:.1f}m", flush=True)

    # verify
    print("verifying sha256 ...", flush=True)
    h = hashlib.sha256()
    with open(PART, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    digest = h.hexdigest()
    if digest != SHA256:
        print(f"SHA MISMATCH got={digest} want={SHA256}", flush=True)
        sys.exit(1)
    print("sha256 OK", flush=True)

    os.replace(PART, BLOB)
    if os.path.exists(DONE_MARK):
        os.remove(DONE_MARK)
    # remove any stale .incomplete
    inc = BLOB + ".incomplete"
    if os.path.exists(inc):
        os.remove(inc)

    # create snapshot symlink: checkpoints/model.ckpt -> ../../../blobs/<sha>
    os.makedirs(os.path.dirname(LINK), exist_ok=True)
    rel = os.path.relpath(BLOB, os.path.dirname(LINK))
    if os.path.islink(LINK) or os.path.exists(LINK):
        os.remove(LINK)
    os.symlink(rel, LINK)
    print(f"BLOB ready: {BLOB}", flush=True)
    print(f"SYMLINK: {LINK} -> {rel}", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
