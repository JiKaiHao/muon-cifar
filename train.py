"""Train CIFAR ResNet-18 with SGD, AdamW, or Muon.

Examples:
  python train.py --optimizer muon --epochs 100 --seed 0 --out runs/muon_s0
  python train.py --optimizer adamw --lr 1e-3 --epochs 100 --seed 0 --out runs/adamw_s0
  python train.py --optimizer sgd --lr 0.1 --epochs 100 --seed 0 --out runs/sgd_s0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CIFAR10
from models import count_parameters, resnet18_cifar
from muon_opt import MuonWithAuxAdam, split_muon_params

OPT_DEFAULTS = {
    "sgd": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4},
    "adamw": {"lr": 1e-3, "momentum": 0.9, "weight_decay": 5e-4},
    "muon": {"lr": 1e-3, "momentum": 0.95, "weight_decay": 5e-4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 optimizer comparison")
    parser.add_argument("--optimizer", choices=("sgd", "adamw", "muon"), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--aux-lr", type=float, default=None, help="AdamW lr inside Muon; defaults to --lr")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--adjust-lr", choices=("original", "match_rms_adamw", "none"), default="match_rms_adamw")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=0, help="limit steps per epoch; 0 runs the full epoch")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def learning_rate(epoch: int, total: int, warmup: int, base: float) -> float:
    if warmup > 0 and epoch < warmup:
        return base * float(epoch + 1) / float(warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_lr(optimizer: torch.optim.Optimizer, base_lrs: list[float], scale: float) -> None:
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base * scale


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=False,
        )
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
    muon_params, aux_params = split_muon_params(model)
    aux_lr = args.lr if args.aux_lr is None else args.aux_lr
    return MuonWithAuxAdam(
        [
            {
                "params": muon_params,
                "lr": args.lr,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "ns_steps": args.ns_steps,
                "adjust_lr": args.adjust_lr,
                "nesterov": True,
                "use_muon": True,
            },
            {
                "params": aux_params,
                "lr": aux_lr,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": args.weight_decay,
                "use_muon": False,
            },
        ]
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        total_loss += float(loss)
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total += labels.numel()
    return total_loss / total, total_correct / total


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for step, (images, labels) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * labels.numel()
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total += labels.numel()
    return total_loss / total, total_correct / total


def main() -> None:
    args = parse_args()
    defaults = OPT_DEFAULTS[args.optimizer]
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.momentum is None:
        args.momentum = defaults["momentum"]
    if args.weight_decay is None:
        args.weight_decay = defaults["weight_decay"]

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.backends.cudnn.benchmark = True

    train_set = CIFAR10(args.data, train=True, augment=True)
    test_set = CIFAR10(args.data, train=False, augment=False)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = resnet18_cifar().to(device)
    optimizer = build_optimizer(model, args)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history_path = out / "history.jsonl"

    n_params = count_parameters(model)
    muon_params, aux_params = split_muon_params(model)
    meta = {
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "aux_lr": args.lr if args.aux_lr is None else args.aux_lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "ns_steps": args.ns_steps if args.optimizer == "muon" else None,
        "adjust_lr": args.adjust_lr if args.optimizer == "muon" else None,
        "warmup_epochs": args.warmup_epochs,
        "seed": args.seed,
        "parameters": n_params,
        "muon_parameters": sum(p.numel() for p in muon_params),
        "aux_parameters": sum(p.numel() for p in aux_params),
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)

    best_acc = -1.0
    for epoch in range(args.epochs):
        scale = learning_rate(epoch, args.epochs, args.warmup_epochs, 1.0)
        apply_lr(optimizer, base_lrs, scale)
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, args.max_steps)
        test_loss, test_acc = evaluate(model, test_loader, device)
        elapsed = time.perf_counter() - start
        metrics = {"ortho_residual": None, "update_rms": None}
        if isinstance(optimizer, MuonWithAuxAdam):
            metrics = optimizer.pop_metrics()
        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "ortho_residual": metrics["ortho_residual"],
            "update_rms": metrics["update_rms"],
            "seconds": elapsed,
        }
        with history_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch + 1:03d}/{args.epochs}  lr {row['lr']:.6g}  "
            f"train {train_acc:.4f}  test {test_acc:.4f}  loss {test_loss:.4f}  {elapsed:.1f}s",
            flush=True,
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "test_acc": test_acc}, out / "best.pt")

    print(f"best_test_acc {best_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
