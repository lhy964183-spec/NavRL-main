from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.package_artifacts import package_artifacts


def test_package_artifacts_includes_checkpoint_tensorboard_and_log():
    tmp_path = Path("tmp") / "test_package_artifacts"
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True)
    checkpoint_dir = tmp_path / "checkpoints"
    tensorboard_dir = tmp_path / "tensorboard"
    checkpoint_dir.mkdir()
    tensorboard_dir.mkdir()
    (checkpoint_dir / "checkpoint_final.pt").write_bytes(b"model")
    (tensorboard_dir / "events.out.tfevents.test").write_bytes(b"events")
    log_file = tmp_path / "train.log"
    log_file.write_text("done\n", encoding="utf-8")
    output = tmp_path / "artifacts.zip"

    package_artifacts(
        output=output,
        checkpoint_dir=checkpoint_dir,
        tensorboard_dir=tensorboard_dir,
        log_file=log_file,
    )

    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "checkpoints/checkpoint_final.pt",
            "tensorboard/events.out.tfevents.test",
            "train.log",
        }
    shutil.rmtree(tmp_path, ignore_errors=True)
