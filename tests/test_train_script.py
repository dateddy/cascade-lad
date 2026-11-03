import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_train_creates_run_dir(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "experiment=dummy",
            "run_name=t",
            f"paths.runs_dir={tmp_path.as_posix()}",
        ],
        cwd=ROOT,
        check=True,
    )
    run_dir = tmp_path / "t"
    for name in ("config.yaml", "git_commit.txt", "run.log"):
        assert (run_dir / name).is_file()
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "[setup]" in log
    assert "config: configs/experiment/dummy.yaml" in log
    assert re.search(r"git_sha: (\w+) \((clean|dirty)\)", log)
    assert not (ROOT / "outputs").exists()
