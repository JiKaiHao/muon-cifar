"""CIFAR-10 loader. Avoids torchvision so the project runs on a plain PyTorch install."""

from __future__ import annotations

import pickle
import tarfile
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _report_download(block_index: int, block_size: int, total: int) -> None:
    if total <= 0 or block_index % 80 != 0:
        return
    got = min(total, block_index * block_size)
    print(f"\rCIFAR-10  {got / 1e6:.1f}/{total / 1e6:.1f} MB", end="", flush=True)


def download_cifar10(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    extracted = root / "cifar-10-batches-py"
    if extracted.is_dir():
        return extracted
    archive_path = root / "cifar-10-python.tar.gz"
    if not _archive_ok(archive_path):
        partial = archive_path.with_suffix(".tar.gz.partial")
        print(f"downloading CIFAR-10 to {archive_path}", flush=True)
        urllib.request.urlretrieve(CIFAR_URL, partial, reporthook=_report_download)
        print(flush=True)
        partial.replace(archive_path)
    if not _archive_ok(archive_path):
        raise RuntimeError(f"CIFAR-10 archive is incomplete: {archive_path}")
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(root)
    if not extracted.is_dir():
        raise FileNotFoundError(f"CIFAR-10 extraction failed under {root}")
    return extracted


def _archive_ok(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 160_000_000:
        return False
    try:
        with tarfile.open(path, "r:gz") as archive:
            return "cifar-10-batches-py/data_batch_1" in archive.getnames()
    except (tarfile.TarError, EOFError, OSError):
        return False


class CIFAR10(Dataset):
    def __init__(self, root: str | Path, train: bool = True, download: bool = True, augment: bool = False):
        folder = download_cifar10(Path(root)) if download else Path(root) / "cifar-10-batches-py"
        names = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        images = []
        labels = []
        for name in names:
            with open(folder / name, "rb") as handle:
                blob = pickle.load(handle, encoding="latin1")
            raw = blob["data"]
            images.append(torch.tensor(raw, dtype=torch.uint8).reshape(-1, 3, 32, 32))
            labels.append(torch.tensor(blob["labels"], dtype=torch.long))
        self.images = torch.cat(images, dim=0)
        self.labels = torch.cat(labels, dim=0)
        self.augment = augment
        self.mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
        self.std = torch.tensor(CIFAR_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        if self.augment:
            image = _augment(image)
        tensor = image.float() / 255.0
        tensor = (tensor - self.mean) / self.std
        return tensor, int(self.labels[index])


def _augment(image: torch.Tensor) -> torch.Tensor:
    """Pad by 4, random 32 crop, random horizontal flip. ``image`` is uint8 CHW."""
    padded = torch.nn.functional.pad(image, (4, 4, 4, 4))
    top = int(torch.randint(0, 9, ()))
    left = int(torch.randint(0, 9, ()))
    cropped = padded[:, top : top + 32, left : left + 32]
    if float(torch.rand(())) < 0.5:
        cropped = cropped.flip(-1)
    return cropped
