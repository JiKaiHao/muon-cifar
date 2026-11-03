"""Plot test accuracy curves from runs/*/history.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_run(path: Path) -> tuple[dict, list[dict]]:
    config = json.loads((path / "config.json").read_text())
    rows = [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines() if line.strip()]
    return config, rows


def label_of(path: Path, config: dict) -> str:
    opt = config["optimizer"]
    if opt == "muon":
        return f"Muon ns={config['ns_steps']}"
    return opt.upper() if opt == "sgd" else "AdamW"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, default="runs")
    parser.add_argument("--out", type=str, default="runs/test_acc.png")
    args = parser.parse_args()
    root = Path(args.runs)
    curves = []
    for path in sorted(p for p in root.iterdir() if (p / "history.jsonl").is_file()):
        config, rows = load_run(path)
        if not rows:
            continue
        curves.append((label_of(path, config) + f" seed {config['seed']}", rows))
    if not curves:
        raise SystemExit(f"no finished runs under {root}")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for name, rows in curves:
        epochs = [row["epoch"] for row in rows]
        acc = [100.0 * row["test_acc"] for row in rows]
        ax.plot(epochs, acc, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
