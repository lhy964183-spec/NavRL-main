import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "isaac-training" / "training" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from tensorboard_utils import log_scalars, scalar_items


class RecordingWriter:
    def __init__(self):
        self.calls = []

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, value, step))


class ScalarLike:
    def numel(self):
        return 1

    def item(self):
        return 2.5


def test_scalar_items_accepts_numbers_and_scalar_tensors():
    values = dict(scalar_items({"loss": 1.5, "count": 2, "tensor": ScalarLike()}))

    assert values == {"loss": 1.5, "count": 2.0, "tensor": 2.5}


def test_scalar_items_skips_non_scalars_and_non_numeric_values():
    values = dict(scalar_items({"vector": [1, 2], "text": "skip", "nan": math.nan}))

    assert "vector" not in values
    assert "text" not in values
    assert math.isnan(values["nan"])


def test_log_scalars_normalizes_tags_and_writes_step():
    writer = RecordingWriter()

    log_scalars(writer, {"train/loss": 0.25, "nested": {"ignored": 1}}, step=7)

    assert writer.calls == [("train/loss", 0.25, 7)]
