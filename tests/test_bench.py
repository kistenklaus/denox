from helpers import run_denox
from models import MODELS;

def test_bench_db(cache_dir, tmp_path):
    onnx_path = tmp_path / "net.onnx"
    db_path = cache_dir / "gpu.db"

    for onnx in MODELS:
        onnx.save(onnx_path)
        output = run_denox(
            "populate", 
            db_path,
            onnx_path,
            verbose=True,
        )
        assert output.returncode == 0
        output = run_denox(
            "bench", 
            db_path,
            "--min-samples=2",
            "--max-samples=2",
            "--relative-error=1",
            verbose=True,
        ) 
        assert output.returncode == 0
        output = run_denox(
            "bench", 
            db_path,
            "--min-samples=2",
            "--max-samples=2",
            "--relative-error=1",
            verbose=True,
        ) 
        assert output.returncode == 0

def test_compile(cache_dir, tmp_path):
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    db_path = cache_dir / "gpu.db"

    for onnx in MODELS:
        onnx.save(onnx_path)
        output = run_denox(
            "compile", onnx_path,
            f"--db={db_path}",
            "-o", dnx_path,
            "--min-samples=2",
            "--max-samples=2",
            "--relative-error=1",
            "--shape", "input=H:W:C",
            verbose=True,
        )
        assert output.returncode == 0

        output = run_denox("bench",
                  dnx_path)
        assert output.returncode == 0

        output = run_denox("bench",
                  dnx_path,
                  "--spec", "H=512", "W=512");
        assert output.returncode == 0
