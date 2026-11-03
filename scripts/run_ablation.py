"""Newton–Schulz step ablation at the Muon learning rate from the main comparison.

Steps 1, 2 and 3 are trained here. Step 5 is the main Muon run and is not repeated.
One job per GPU. A run with 100 logged epochs is skipped.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from queue import Queue
from threading import Thread

EPOCHS = 100
WARMUP = 5
LR = 3e-3
STEPS = (1, 2, 3)
SEEDS = (0, 1, 2)


def run(steps: int, seed: int, gpu: str) -> None:
    name = f"ns{steps}_s{seed}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = [
        sys.executable,
        "train.py",
        "--optimizer",
        "muon",
        "--lr",
        str(LR),
        "--epochs",
        str(EPOCHS),
        "--warmup-epochs",
        str(WARMUP),
        "--ns-steps",
        str(steps),
        "--adjust-lr",
        "match_rms_adamw",
        "--seed",
        str(seed),
        "--workers",
        "4",
        "--out",
        f"runs/ablation/{name}",
        "--device",
        "cuda",
    ]
    print(f"start {name} on gpu {gpu}", flush=True)
    subprocess.run(cmd, check=True, env=env)
    print(f"done {name}", flush=True)


def is_finished(steps: int, seed: int) -> bool:
    history = Path("runs/ablation") / f"ns{steps}_s{seed}" / "history.jsonl"
    if not history.is_file():
        return False
    lines = [line for line in history.read_text().splitlines() if line.strip()]
    return len(lines) >= EPOCHS


def main() -> None:
    pending: Queue[tuple[int, int] | None] = Queue()
    gpus: Queue[str] = Queue()
    for gpu in ("0", "1"):
        gpus.put(gpu)
    queued = 0
    for steps in STEPS:
        for seed in SEEDS:
            if is_finished(steps, seed):
                print(f"skip ns{steps}_s{seed}", flush=True)
                continue
            pending.put((steps, seed))
            queued += 1
    print(f"queued {queued} jobs  lr={LR}  steps={STEPS}", flush=True)

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

    threads = [Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for _ in threads:
        pending.put(None)
    pending.join()
    for thread in threads:
        thread.join()
    print("ablation finished", flush=True)


if __name__ == "__main__":
    main()
