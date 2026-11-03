"""Short learning-rate sweep. One job per GPU.

CIFAR ResNet-18 at batch 128 reports a low utilization percentage, but stacking
four jobs on one A40 only filled that counter: each epoch slowed from about 11s
to about 50s, so the sweep did not finish sooner. Finished runs are skipped.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from queue import Queue
from threading import Thread

EPOCHS = 15
SLOTS_PER_GPU = 1

JOBS = [
    ("sgd", 0.05, "sgd_lr0.05"),
    ("sgd", 0.1, "sgd_lr0.1"),
    ("sgd", 0.2, "sgd_lr0.2"),
    ("adamw", 3e-4, "adamw_lr3e-4"),
    ("adamw", 1e-3, "adamw_lr1e-3"),
    ("adamw", 3e-3, "adamw_lr3e-3"),
    ("muon", 3e-4, "muon_lr3e-4"),
    ("muon", 1e-3, "muon_lr1e-3"),
    ("muon", 3e-3, "muon_lr3e-3"),
]


def run(opt: str, lr: float, name: str, gpu: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = [
        sys.executable,
        "train.py",
        "--optimizer",
        opt,
        "--lr",
        str(lr),
        "--epochs",
        "15",
        "--warmup-epochs",
        "1",
        "--seed",
        "0",
        "--workers",
        "4",
        "--out",
        f"runs/sweep/{name}",
        "--device",
        "cuda",
    ]
    print(f"start {name} on gpu {gpu}", flush=True)
    subprocess.run(cmd, check=True, env=env)
    print(f"done {name}", flush=True)


def is_finished(name: str) -> bool:
    history = Path("runs/sweep") / name / "history.jsonl"
    if not history.is_file():
        return False
    lines = [line for line in history.read_text().splitlines() if line.strip()]
    return len(lines) >= EPOCHS


def main() -> None:
    pending: Queue[tuple[str, float, str] | None] = Queue()
    gpus: Queue[str] = Queue()
    for gpu in ("0", "1"):
        for _ in range(SLOTS_PER_GPU):
            gpus.put(gpu)
    queued = 0
    for job in JOBS:
        name = job[2]
        if is_finished(name):
            print(f"skip {name}", flush=True)
            continue
        partial = Path("runs/sweep") / name
        if partial.exists():
            shutil.rmtree(partial)
        pending.put(job)
        queued += 1
    print(f"queued {queued} jobs, {SLOTS_PER_GPU} per GPU", flush=True)

    def worker() -> None:
        while True:
            job = pending.get()
            if job is None:
                pending.task_done()
                return
            gpu = gpus.get()
            try:
                run(*job, gpu)
            finally:
                gpus.put(gpu)
                pending.task_done()

    threads = [Thread(target=worker) for _ in range(SLOTS_PER_GPU * 2)]
    for thread in threads:
        thread.start()
    for _ in threads:
        pending.put(None)
    pending.join()
    for thread in threads:
        thread.join()
    print("sweep finished", flush=True)


if __name__ == "__main__":
    main()
