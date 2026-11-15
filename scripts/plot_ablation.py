"""Figures for the Newton–Schulz step ablation.

Two panels, both against the number of Newton–Schulz steps:
  left  – mean orthogonality residual at the last epoch (lower is "more polar");
  right – test accuracy, mean ± sd over seeds.

Step 5 reuses the main Muon runs, matching the protocol in run_ablation.py.

  python scripts/plot_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SEEDS = (0, 1, 2)
STEPS = (1, 2, 3, 5)


def run_dirs(steps: int) -> list[Path]:
    if steps == 5:
        return [RUNS / "main" / f"muon_s{seed}" for seed in SEEDS]
    return [RUNS / "ablation" / f"ns{steps}_s{seed}" for seed in SEEDS]


def load(path: Path) -> list[dict]:
    history = path / "history.jsonl"
    if not history.is_file():
        return []
    return [json.loads(line) for line in history.read_text().splitlines() if line.strip()]


def collect() -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    steps_used, residual_mean, residual_sd, acc_mean, acc_sd = [], [], [], [], []
    for steps in STEPS:
        residuals, accs = [], []
        for path in run_dirs(steps):
            rows = load(path)
            if not rows or rows[-1].get("ortho_residual") is None:
                continue
            residuals.append(rows[-1]["ortho_residual"])
            accs.append(100.0 * rows[-1]["test_acc"])
        if not residuals:
            continue
        steps_used.append(steps)
        residual_mean.append(mean(residuals))
        residual_sd.append(stdev(residuals) if len(residuals) > 1 else 0.0)
        acc_mean.append(mean(accs))
        acc_sd.append(stdev(accs) if len(accs) > 1 else 0.0)
    return steps_used, residual_mean, residual_sd, acc_mean, acc_sd


def main() -> None:
    steps, r_mean, r_sd, a_mean, a_sd = collect()
    if not steps:
        raise SystemExit("no ablation runs finished yet")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    ax.errorbar(steps, r_mean, yerr=r_sd, marker="o", capsize=3, color="#b5651d")
    ax.set_xlabel("Newton–Schulz steps")
    ax.set_ylabel("Orthogonality residual (last epoch)")
    ax.set_xticks(steps)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.errorbar(steps, a_mean, yerr=a_sd, marker="o", capsize=3, color="#1f4e79")
    ax.set_xlabel("Newton–Schulz steps")
    ax.set_ylabel("Test accuracy (%), 100 epochs")
    ax.set_xticks(steps)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = RUNS / "ablation" / "steps_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print(out)
    for x, rm, rs, am, asd in zip(steps, r_mean, r_sd, a_mean, a_sd):
        print(f"steps={x}  residual={rm:.4f}±{rs:.4f}  test={am:.2f}±{asd:.2f}%")


if __name__ == "__main__":
    main()
