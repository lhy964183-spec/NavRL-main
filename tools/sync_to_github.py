#!/usr/bin/env python3
"""Commit and push the current NavRL working tree to GitHub.

Authentication is delegated to the user's normal Git credential helper or SSH
key. The script never accepts or stores a token, and it does not force-push.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Sequence


def build_sync_commands(
    remote: str,
    branch: str,
    message: str,
    repo_url: str | None = None,
) -> list[list[str]]:
    """Build the non-destructive Git command sequence used by ``main``."""

    commands: list[list[str]] = []
    if repo_url:
        commands.append(["git", "remote", "add", remote, repo_url])
    commands.extend(
        [
            ["git", "add", "-A"],
            ["git", "commit", "-m", message],
            ["git", "push", "-u", remote, branch],
        ]
    )
    return commands


def run_git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=check,
        text=True,
    )


def git_output(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the project containing this script).",
    )
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--repo-url", help="Add this URL when the remote is not configured.")
    parser.add_argument("--branch", help="Branch to push; defaults to the current branch or main.")
    parser.add_argument("--message", help="Commit message.")
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize a Git repository when the project is not already one.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()

    is_repo = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not is_repo:
        if not args.init:
            print(
                "No Git repository found. Run with --init --repo-url <GitHub URL> "
                "or initialize this directory manually.",
                file=sys.stderr,
            )
            return 2
        run_git(repo_root, "init")

    current_branch = ""
    branch_result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    current_branch = branch_result.stdout.strip()
    branch = args.branch or current_branch or "main"
    if not current_branch:
        run_git(repo_root, "checkout", "-B", branch)

    remote_url = subprocess.run(
        ["git", "remote", "get-url", args.remote],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.strip()
    if not remote_url:
        if not args.repo_url:
            print(
                f"Remote '{args.remote}' is not configured. Supply --repo-url.",
                file=sys.stderr,
            )
            return 2
        run_git(repo_root, "remote", "add", args.remote, args.repo_url)

    run_git(repo_root, "add", "-A")
    staged_changes = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo_root,
        check=False,
    ).returncode != 0
    if not staged_changes:
        print("No changes to commit.")
        return 0

    message = args.message or f"Update NavRL {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}"
    run_git(repo_root, "commit", "-m", message)
    run_git(repo_root, "push", "-u", args.remote, branch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
