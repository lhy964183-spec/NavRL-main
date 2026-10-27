#!/usr/bin/env python3
"""Run NavRL's Isaac Sim evaluator against a downloaded checkpoint.

This wrapper keeps the original evaluator as the source of truth while making
the checkpoint path and evaluation budget configurable from a local machine.
It must be run in an environment where Isaac Sim and the NavRL dependencies
are installed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


def build_command(
    repo_root: Path,
    checkpoint: Path,
    max_frames: int,
    num_envs: int | None,
    device: str,
    extra_overrides: Sequence[str] = (),
) -> list[str]:
    """Build the evaluator command without invoking a shell."""

    eval_script = repo_root / "isaac-training" / "training" / "scripts" / "eval.py"
    command = [
        sys.executable,
        eval_script.as_posix(),
        f"checkpoint={checkpoint.resolve().as_posix()}",
        "headless=True",
        f"max_frame_num={int(max_frames)}",
        f"device={device}",
        "wandb.mode=offline",
        "tensorboard.enabled=False",
    ]
    if num_envs is not None:
        command.append(f"env.num_envs={int(num_envs)}")
    command.extend(extra_overrides)
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Local NavRL repository root (defaults to this script's parent).",
    )
    parser.add_argument("--max-frames", type=int, default=4096)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override; repeat for multiple values.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    command = build_command(
        repo_root=repo_root,
        checkpoint=checkpoint,
        max_frames=args.max_frames,
        num_envs=args.num_envs,
        device=args.device,
        extra_overrides=args.override,
    )
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.run(command, cwd=repo_root / "isaac-training", env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
