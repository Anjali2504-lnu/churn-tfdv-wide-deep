import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_data_validation_runs():
    out_dir = ROOT / "artifacts_test" / "tfdv"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT/"src"/"data_validation.py"), "--data_dir", str(ROOT/"data"), "--out_dir", str(out_dir)]
    subprocess.check_call(cmd)

def test_training_runs_short():
    out_dir = ROOT / "artifacts_test" / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT/"src"/"train_model.py"), "--data_dir", str(ROOT/"data"), "--out_dir", str(out_dir), "--epochs", "1", "--batch_size", "128"]
    subprocess.check_call(cmd)
