"""Download the CIFAR-10 archive in several HTTP ranges. The Toronto host is slow per connection."""

from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
SIZE = 170_498_071
PARTS = 8
ROOT = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    part_dir = ROOT / "parts"
    part_dir.mkdir(exist_ok=True)
    dest = ROOT / "cifar-10-python.tar.gz"
    if dest.is_file() and dest.stat().st_size == SIZE and _archive_ok(dest):
        print("archive already complete", flush=True)
    else:
        chunk = (SIZE + PARTS - 1) // PARTS
        procs = []
        for index in range(PARTS):
            start = index * chunk
            end = min(SIZE - 1, (index + 1) * chunk - 1)
            out = part_dir / f"part.{index:02d}"
            expected = end - start + 1
            if out.is_file() and out.stat().st_size == expected:
                print(f"part {index} already downloaded", flush=True)
                continue
            cmd = [
                "curl",
                "-L",
                "--retry",
                "30",
                "--retry-delay",
                "2",
                "-C",
                "-",
                "-r",
                f"{start}-{end}",
                "-o",
                str(out),
                URL,
            ]
            procs.append(subprocess.Popen(cmd))
        codes = [proc.wait() for proc in procs]
        print("curl_exit", codes, flush=True)
        if any(code != 0 for code in codes):
            raise SystemExit(1)
        with dest.open("wb") as handle:
            for index in range(PARTS):
                handle.write((part_dir / f"part.{index:02d}").read_bytes())
        print("bytes", dest.stat().st_size, flush=True)
    if not _archive_ok(dest):
        raise SystemExit("archive failed the gzip check")
    with tarfile.open(dest, "r:gz") as archive:
        archive.extractall(ROOT)
    print("extracted", ROOT / "cifar-10-batches-py", flush=True)


def _archive_ok(path: Path) -> bool:
    try:
        with tarfile.open(path, "r:gz") as archive:
            return "cifar-10-batches-py/data_batch_1" in archive.getnames()
    except (tarfile.TarError, EOFError, OSError):
        return False


if __name__ == "__main__":
    main()
