from helpers import run_denox
from models import MODELS;
from images import create_random_png
from inference import onnx_infer, load_output_image
import torch


def test_populate(cache_dir, tmp_path):
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
            "populate", 
            db_path,
            onnx_path,
            verbose=True,
        ) 
        assert output.returncode == 0

def test_populate_with_device_yml(cache_dir, tmp_path):
    onnx_path = tmp_path / "net.onnx"
    db_path = cache_dir / "gpu.db"

    device_yml = tmp_path / "gpu.yaml";

    output = run_denox("query-device-info", "-o", device_yml);
    assert output.returncode == 0

    for onnx in MODELS:
        onnx.save(onnx_path)
        output = run_denox(
            "populate", 
            db_path,
            onnx_path,
            "--device", device_yml,
            verbose=True,
        )
        assert output.returncode == 0
        output = run_denox(
            "populate", 
            db_path,
            onnx_path,
            "--device", device_yml,
            verbose=True,
        ) 
        assert output.returncode == 0
