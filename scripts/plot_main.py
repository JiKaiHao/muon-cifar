"""Mean test-accuracy curves for the 100-epoch comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "main"
LABELS = {"sgd": "SGD", "adamw": "AdamW", "muon": "Muon"}


def load(opt: str, seed: int) -> list[dict]:
    path = RUNS / f"{opt}_s{seed}" / "history.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for opt in ("sgd", "adamw", "muon"):
        curves = []
        for seed in (0, 1, 2):
            rows = load(opt, seed)
            curves.append([100.0 * row["test_acc"] for row in rows])
        arr = np.array(curves)
        epochs = np.arange(1, arr.shape[1] + 1)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        line, = ax.plot(epochs, mean, label=LABELS[opt])
        ax.fill_between(epochs, mean - std, mean + std, color=line.get_color(), alpha=0.18)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = RUNS / "test_acc.png"
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
