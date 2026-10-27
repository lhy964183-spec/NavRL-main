#!/usr/bin/env python3
"""Package NavRL checkpoints, TensorBoard events, and training logs."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


def _add_path(archive: zipfile.ZipFile, source: Path, archive_root: str, seen: set[str]) -> int:
    if not source.exists():
        raise FileNotFoundError(f"Artifact path does not exist: {source}")
    paths = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
    count = 0
    for path in paths:
        relative = path.name if source.is_file() else path.relative_to(source).as_posix()
        member = f"{archive_root}/{relative}" if archive_root else relative
        if member in seen:
            continue
        archive.write(path, member)
        seen.add(member)
        count += 1
    return count


def package_artifacts(
    output: Path,
    checkpoint_dir: Path,
    tensorboard_dir: Path,
    log_file: Path | None = None,
) -> Path:
    """Create a zip archive and return its path."""

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _add_path(archive, checkpoint_dir.expanduser().resolve(), "checkpoints", seen)
        _add_path(archive, tensorboard_dir.expanduser().resolve(), "tensorboard", seen)
        if log_file is not None:
            _add_path(archive, log_file.expanduser().resolve(), "", seen)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = package_artifacts(
        output=args.output,
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_dir=args.tensorboard_dir,
        log_file=args.log_file,
    )
    print(f"Wrote artifact archive: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
