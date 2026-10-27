from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.validate_checkpoint import build_command


def test_build_command_passes_checkpoint_and_runtime_overrides():
    command = build_command(
        repo_root=Path("D:/NavRL"),
        checkpoint=Path("D:/runs/checkpoint_final.pt"),
        max_frames=256,
        num_envs=4,
        device="cuda:0",
        extra_overrides=["env.num_obstacles=0"],
    )

    assert command[1].endswith("isaac-training/training/scripts/eval.py")
    assert "checkpoint=D:/runs/checkpoint_final.pt" in command
    assert "max_frame_num=256" in command
    assert "env.num_envs=4" in command
    assert "device=cuda:0" in command
    assert "env.num_obstacles=0" in command
    assert "wandb.mode=offline" in command
