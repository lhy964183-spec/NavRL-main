from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.sync_to_github import build_sync_commands


def test_build_sync_commands_adds_remote_commits_and_pushes_branch():
    commands = build_sync_commands(
        remote="origin",
        branch="main",
        message="update training workflow",
        repo_url="git@github.com:example/NavRL.git",
    )

    assert commands == [
        ["git", "remote", "add", "origin", "git@github.com:example/NavRL.git"],
        ["git", "add", "-A"],
        ["git", "commit", "-m", "update training workflow"],
        ["git", "push", "-u", "origin", "main"],
    ]
