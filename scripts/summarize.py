"""Summarise finished runs into markdown tables.

Reading conventions, fixed once so the report tables are reproducible:
  * "best"  = highest test accuracy over the logged epochs (that is what
              train.py writes to best.pt, and what the 15-epoch sweep uses).
  * "last"  = test accuracy of the final logged epoch (unbiased, no selection).
  * wall    = sum of the per-epoch "seconds" field, i.e. training + evaluation.
  * ortho   = mean orthogonality residual over the last epoch of the run.

  python scripts/summarize.py                       # writes runs/summary.md
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SEEDS = (0, 1, 2)


def load(run_dir: Path) -> tuple[dict, list[dict]] | tuple[None, None]:
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.jsonl"
    if not config_path.is_file() or not history_path.is_file():
        return None, None
    config = json.loads(config_path.read_text())
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    return config, rows


def best_acc(rows: list[dict]) -> float:
    return max(row["test_acc"] for row in rows)


def last_acc(rows: list[dict]) -> float:
    return rows[-1]["test_acc"]


def wall_minutes(rows: list[dict]) -> float:
    return sum(row["seconds"] for row in rows) / 60.0


def last_ortho(rows: list[dict]) -> float | None:
    values = [row["ortho_residual"] for row in rows if row.get("ortho_residual") is not None]
    return values[-1] if values else None


def fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def mean_std(values: list[float]) -> str:
    """Format accuracies given as fractions: percent mean plus percent sd."""
    if not values:
        return "—"
    if len(values) == 1:
        return pct(values[0])
    return f"{pct(mean(values))} ± {100.0 * stdev(values):.2f}"


def sweep_section(lines: list[str]) -> None:
    lines.append("## 学习率扫描（15 epoch，seed 0）\n")
    lines.append("| 优化器 | 学习率 | best test | last test | 墙钟(min) |")
    lines.append("| --- | --- | --- | --- | --- |")
    best: dict[str, tuple[float, float]] = {}
    for opt in ("sgd", "adamw", "muon"):
        for lr_tag in ("0.05", "0.1", "0.2", "3e-4", "1e-3", "3e-3"):
            name = f"{opt}_lr{lr_tag}"
            config, rows = load(RUNS / "sweep" / name)
            if not config or len(rows) < config["epochs"]:
                continue
            acc = best_acc(rows)
            lr = config["lr"]
            lr_text = f"{lr:.0e}" if lr < 0.01 else f"{lr:g}"
            lines.append(
                f"| {opt} | {lr_text} "
                f"| {pct(acc)} | {pct(last_acc(rows))} | {fmt(wall_minutes(rows), 1)} |"
            )
            if opt not in best or acc > best[opt][1]:
                best[opt] = (config["lr"], acc)
    lines.append("")
    if best:
        lines.append("按 best test 选中的学习率：" + "，".join(
            f"{opt} = {lr:g}（{pct(acc)}）" for opt, (lr, acc) in best.items()
        ) + "\n")


def main_section(lines: list[str]) -> None:
    lines.append("## 正式对比（100 epoch，3 个种子）\n")
    lines.append("| 优化器 | 学习率 | best test (mean ± sd) | last test (mean ± sd) |"
                 " 每 epoch (s) | 墙钟 (min) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for opt in ("sgd", "adamw", "muon"):
        bests, lasts, per_epoch, walls = [], [], [], []
        lr = None
        for seed in SEEDS:
            config, rows = load(RUNS / "main" / f"{opt}_s{seed}")
            if not config or len(rows) < config["epochs"]:
                continue
            lr = config["lr"]
            bests.append(best_acc(rows))
            lasts.append(last_acc(rows))
            per_epoch.append(mean([row["seconds"] for row in rows]))
            walls.append(wall_minutes(rows))
        label = {"sgd": "SGD", "adamw": "AdamW", "muon": "Muon"}[opt]
        if not bests:
            lines.append(f"| {label} | — | 未完成 | 未完成 | — | — |")
            continue
        lines.append(
            f"| {label} | {lr:g} | {mean_std(bests)} | {mean_std(lasts)} "
            f"| {fmt(mean(per_epoch), 1)} | {fmt(mean(walls), 1)} |"
        )
    lines.append("")


def sgd_lr_section(lines: list[str]) -> None:
    """SGD at the sweep's two best learning rates, 100 epochs each.

    The 15-epoch sweep separated lr=0.05 and lr=0.1 by 0.3 points at one seed,
    which is inside noise, so the 100-epoch comparison is reported for both.
    """
    lines.append("## SGD 两个学习率的 100-epoch 对照\n")
    lines.append("| 学习率 | best test (mean ± sd) | last test (mean ± sd) | 每 epoch (s) | 墙钟 (min) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for lr, root in ((0.05, RUNS / "main"), (0.1, RUNS / "sgd_lr01")):
        bests, lasts, per_epoch, walls = [], [], [], []
        for seed in SEEDS:
            config, rows = load(root / f"sgd_s{seed}")
            if not config or len(rows) < config["epochs"]:
                continue
            bests.append(best_acc(rows))
            lasts.append(last_acc(rows))
            per_epoch.append(mean([row["seconds"] for row in rows]))
            walls.append(wall_minutes(rows))
        if not bests:
            lines.append(f"| {lr:g} | 未完成 | 未完成 | — | — |")
            continue
        lines.append(
            f"| {lr:g} | {mean_std(bests)} | {mean_std(lasts)} "
            f"| {fmt(mean(per_epoch), 1)} | {fmt(mean(walls), 1)} |"
        )
    lines.append("")


def ablation_section(lines: list[str]) -> None:
    lines.append("## Newton–Schulz 步数消融（100 epoch，学习率 3e-3，3 个种子）\n")
    lines.append("| ns_steps | best test (mean ± sd) | last test (mean ± sd) | 末轮正交残差 |"
                 " 末轮更新 RMS | 每 epoch (s) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for steps in (1, 2, 3, 5):
        if steps == 5:
            dirs = [RUNS / "main" / f"muon_s{seed}" for seed in SEEDS]
        else:
            dirs = [RUNS / "ablation" / f"ns{steps}_s{seed}" for seed in SEEDS]
        bests, lasts, orthos, rmss, per_epoch = [], [], [], [], []
        for run_dir in dirs:
            config, rows = load(run_dir)
            if not config or len(rows) < config["epochs"]:
                continue
            bests.append(best_acc(rows))
            lasts.append(last_acc(rows))
            if last_ortho(rows) is not None:
                orthos.append(last_ortho(rows))
                rmss.append(rows[-1]["update_rms"])
            per_epoch.append(mean([row["seconds"] for row in rows]))
        if not bests:
            lines.append(f"| {steps} | 未完成 | 未完成 | — | — | — |")
            continue
        note = "（来自 main）" if steps == 5 else ""
        lines.append(
            f"| {steps}{note} | {mean_std(bests)} | {mean_std(lasts)} "
            f"| {fmt(mean(orthos) if orthos else None, 4)} "
            f"| {fmt(mean(rmss) if rmss else None, 4)} | {fmt(mean(per_epoch), 1)} |"
        )
    lines.append("")


def main() -> None:
    lines = ["# 实验结果汇总\n",
             f"运行目录：`{RUNS}`；三种子为 seed 0/1/2，批量 128，5 epoch 线性 warmup + cosine。\n"]
    sweep_section(lines)
    main_section(lines)
    sgd_lr_section(lines)
    ablation_section(lines)
    text = "\n".join(lines)
    out = RUNS / "summary.md"
    out.write_text(text + "\n")
    print(text)
    print(f"written {out}")


if __name__ == "__main__":
    main()
