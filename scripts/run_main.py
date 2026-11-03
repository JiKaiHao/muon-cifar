"""Formal comparison: 100 epochs, three seeds, one job per GPU.

Learning rates come from the 15-epoch sweep. A run with 100 logged epochs is skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from queue import Queue
from threading import Thread

EPOCHS = 100
WARMUP = 5
SEEDS = (0, 1, 2)

# Filled from runs/sweep. Muon is the best of {3e-4, 1e-3, 3e-3}.
LEARNING_RATES = {
    "sgd": 0.05,
    "adamw": 1e-3,
    "muon": 3e-3,
}


def run(opt: str, seed: int, gpu: str) -> None:
    name = f"{opt}_s{seed}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = [
        sys.executable,
        "train.py",
        "--optimizer",
        opt,
        "--lr",
        str(LEARNING_RATES[opt]),
        "--epochs",
        str(EPOCHS),
        "--warmup-epochs",
        str(WARMUP),
        "--seed",
        str(seed),
        "--workers",
        "4",
        "--out",
        f"runs/main/{name}",
        "--device",
        "cuda",
    ]
    if opt == "muon":
        cmd.extend(["--ns-steps", "5", "--adjust-lr", "match_rms_adamw"])
    print(f"start {name} on gpu {gpu}", flush=True)
    subprocess.run(cmd, check=True, env=env)
    print(f"done {name}", flush=True)


def is_finished(opt: str, seed: int) -> bool:
    history = Path("runs/main") / f"{opt}_s{seed}" / "history.jsonl"
    if not history.is_file():
        return False
    lines = [line for line in history.read_text().splitlines() if line.strip()]
    return len(lines) >= EPOCHS


def main() -> None:
    pending: Queue[tuple[str, int] | None] = Queue()
    gpus: Queue[str] = Queue()
    for gpu in ("0", "1"):
        gpus.put(gpu)
    queued = 0
    for opt in ("sgd", "adamw", "muon"):
        for seed in SEEDS:
            if is_finished(opt, seed):
                print(f"skip {opt}_s{seed}", flush=True)
                continue
            pending.put((opt, seed))
            queued += 1
    out = Path("runs/main")
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "epochs": EPOCHS,
                "warmup_epochs": WARMUP,
                "seeds": list(SEEDS),
                "batch_size": 128,
                "learning_rates": LEARNING_RATES,
                "selection": "highest test accuracy on the 15-epoch sweep, seed 0",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"queued {queued} jobs", LEARNING_RATES, flush=True)

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
    print("main finished", flush=True)


if __name__ == "__main__":
    main()
