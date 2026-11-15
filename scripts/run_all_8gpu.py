"""Run sweep, main comparison, and Newton–Schulz ablation over eight idle GPUs.

The protocol matches scripts/sweep_lr.py, scripts/run_main.py and
scripts/run_ablation.py exactly; this file only changes the scheduling so the
whole matrix fits the eight local cards instead of two. Finished runs are
skipped, so the script is safe to restart. Per-job stdout goes to
runs/_launcher/<name>.log.

  python scripts/run_all_8gpu.py                      # everything
  python scripts/run_all_8gpu.py --stages main ablation
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]
GPUS = tuple(str(i) for i in range(8))

# Stage 1: 15 epochs, one seed, three learning rates per optimizer.
SWEEP_EPOCHS = 15
SWEEP_JOBS = [
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

# Stage 2: 100 epochs, three seeds, learning rate chosen by the sweep.
MAIN_EPOCHS = 100
MAIN_LRS = {"sgd": 0.05, "adamw": 1e-3, "muon": 3e-3}
MAIN_SEEDS = (0, 1, 2)
MAIN_NS_STEPS = 5

# Stage 3: Newton–Schulz steps at the Muon learning rate. Step 5 is stage 2.
ABLATION_EPOCHS = 100
ABLATION_LR = 3e-3
ABLATION_STEPS = (1, 2, 3)
ABLATION_SEEDS = (0, 1, 2)

# Stage 4: SGD at the second-best sweep learning rate. The 15-epoch sweep put
# lr=0.1 above lr=0.05 by 0.3 points, which is inside seed noise, so both are
# trained for 100 epochs and reported side by side instead of picking one.
SGD_ALT_LR = 0.1

WARMUP_SWEEP = 1
WARMUP_FULL = 5
BATCH_SIZE = 128
WORKERS = 4


def job_command(job: dict) -> list[str]:
    cmd = [
        sys.executable,
        "train.py",
        "--optimizer",
        job["optimizer"],
        "--lr",
        str(job["lr"]),
        "--epochs",
        str(job["epochs"]),
        "--warmup-epochs",
        str(job["warmup"]),
        "--batch-size",
        str(BATCH_SIZE),
        "--seed",
        str(job["seed"]),
        "--workers",
        str(WORKERS),
        "--out",
        job["out"],
        "--device",
        "cuda",
    ]
    if job["optimizer"] == "muon":
        cmd.extend(["--ns-steps", str(job["ns_steps"]), "--adjust-lr", "match_rms_adamw"])
    return cmd


def is_finished(job: dict) -> bool:
    history = ROOT / job["out"] / "history.jsonl"
    if not history.is_file():
        return False
    lines = [line for line in history.read_text().splitlines() if line.strip()]
    return len(lines) >= job["epochs"]


def sweep_jobs() -> list[dict]:
    return [
        {
            "name": name,
            "optimizer": opt,
            "lr": lr,
            "epochs": SWEEP_EPOCHS,
            "warmup": WARMUP_SWEEP,
            "seed": 0,
            "ns_steps": 5,
            "out": f"runs/sweep/{name}",
        }
        for opt, lr, name in SWEEP_JOBS
    ]


def main_jobs() -> list[dict]:
    return [
        {
            "name": f"{opt}_s{seed}",
            "optimizer": opt,
            "lr": MAIN_LRS[opt],
            "epochs": MAIN_EPOCHS,
            "warmup": WARMUP_FULL,
            "seed": seed,
            "ns_steps": MAIN_NS_STEPS,
            "out": f"runs/main/{opt}_s{seed}",
        }
        for opt in ("sgd", "adamw", "muon")
        for seed in MAIN_SEEDS
    ]


def ablation_jobs() -> list[dict]:
    return [
        {
            "name": f"ns{steps}_s{seed}",
            "optimizer": "muon",
            "lr": ABLATION_LR,
            "epochs": ABLATION_EPOCHS,
            "warmup": WARMUP_FULL,
            "seed": seed,
            "ns_steps": steps,
            "out": f"runs/ablation/ns{steps}_s{seed}",
        }
        for steps in ABLATION_STEPS
        for seed in ABLATION_SEEDS
    ]


def sgd_alt_jobs() -> list[dict]:
    return [
        {
            "name": f"sgd_lr01_s{seed}",
            "optimizer": "sgd",
            "lr": SGD_ALT_LR,
            "epochs": MAIN_EPOCHS,
            "warmup": WARMUP_FULL,
            "seed": seed,
            "ns_steps": 5,
            "out": f"runs/sgd_lr01/sgd_s{seed}",
        }
        for seed in MAIN_SEEDS
    ]


STAGES = {
    "sweep": sweep_jobs,
    "main": main_jobs,
    "ablation": ablation_jobs,
    "sgd_lr01": sgd_alt_jobs,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGES), default=["sweep", "main", "ablation"])
    parser.add_argument("--gpus", nargs="+", default=list(GPUS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = ROOT / "runs" / "_launcher"
    log_dir.mkdir(parents=True, exist_ok=True)

    pending: Queue[dict | None] = Queue()
    gpus: Queue[str] = Queue()
    for gpu in args.gpus:
        gpus.put(gpu)

    queued = 0
    for stage in args.stages:
        jobs = STAGES[stage]()
        skipped = [job["name"] for job in jobs if is_finished(job)]
        for name in skipped:
            print(f"skip {name}", flush=True)
        todo = [job for job in jobs if not is_finished(job)]
        print(f"{stage}: {len(todo)} to run, {len(skipped)} finished", flush=True)
        for job in todo:
            job["stage"] = stage
            pending.put(job)
            queued += 1
    if queued == 0:
        print("nothing to do", flush=True)
        return
    print(f"queued {queued} jobs on gpus {args.gpus}", flush=True)

    def worker() -> None:
        while True:
            job = pending.get()
            if job is None:
                pending.task_done()
                return
            gpu = gpus.get()
            log_path = log_dir / f"{job['name']}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            started = time.time()
            try:
                with log_path.open("w") as log:
                    print(f"start {job['name']} on gpu {gpu} -> {log_path}", flush=True)
                    subprocess.run(job_command(job), cwd=ROOT, check=True, env=env, stdout=log, stderr=log)
                minutes = (time.time() - started) / 60.0
                print(f"done {job['name']} ({minutes:.1f} min)", flush=True)
            except subprocess.CalledProcessError as error:
                print(f"FAILED {job['name']} code {error.returncode}, see {log_path}", flush=True)
            finally:
                gpus.put(gpu)
                pending.task_done()

    threads = [Thread(target=worker) for _ in args.gpus]
    for thread in threads:
        thread.start()
    for _ in threads:
        pending.put(None)
    pending.join()
    for thread in threads:
        thread.join()
    print("all finished", flush=True)


if __name__ == "__main__":
    main()
