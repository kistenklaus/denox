from helpers import run_denox
from models import MODELS;
from images import create_random_png
from inference import onnx_infer, load_output_image
import torch

def test_reweight(cache_dir, tmp_path):
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    reweighted_path = tmp_path / "rnet.dnx"
    input_path = tmp_path / "input.png"
    ref_path = tmp_path / "ref.png"
    output_path = tmp_path / "output.png"
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
            verbose=True,
        )
        assert output.returncode == 0
        create_random_png(input_path, 1920, 1080)
        output = run_denox(
            "infer", dnx_path,
            "-i", input_path,
            "-o", ref_path,
            verbose=True,
        )
        assert output.returncode == 0
        output = run_denox(
                "reweight",
                dnx_path,
                onnx_path,
                "-o", reweighted_path)
        assert output.returncode == 0
        
        output = run_denox(
            "infer", reweighted_path,
            "-i", input_path,
            "-o", output_path,
            verbose=True,
        )
        assert output.returncode == 0

        ref = load_output_image(onnx, ref_path)
        image = load_output_image(onnx, output_path)
        if ref is not None:
            assert image is not None
            image = load_output_image(onnx, output_path)
            assert image is not None
            assert ref.shape == image.shape
            torch.testing.assert_close(
                image.to(torch.float32),
                ref.to(torch.float32),
                rtol=1e-1,
                atol=1e-1,
            )


def test_reweight_with_shape_type_and_format(cache_dir, tmp_path):
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    reweighted_path = tmp_path / "rnet.dnx"
    input_path = tmp_path / "input.png"
    ref_path = tmp_path / "ref.png"
    output_path = tmp_path / "output.png"
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
            "--format", "input=hwc",
            "--type", "input=f16",
            verbose=True,
        )
        assert output.returncode == 0
        create_random_png(input_path, 1920, 1080)
        output = run_denox(
            "infer", dnx_path,
            "-i", input_path,
            "-o", ref_path,
            verbose=True,
        )
        assert output.returncode == 0
        output = run_denox(
                "reweight",
                dnx_path,
                onnx_path,
                "-o", reweighted_path)
        assert output.returncode == 0
        
        output = run_denox(
            "infer", reweighted_path,
            "-i", input_path,
            "-o", output_path,
            verbose=True,
        )
        assert output.returncode == 0

        ref = load_output_image(onnx, ref_path)
        image = load_output_image(onnx, output_path)
        if ref is not None:
            assert image is not None
            image = load_output_image(onnx, output_path)
            assert image is not None
            assert ref.shape == image.shape
            torch.testing.assert_close(
                image.to(torch.float32),
                ref.to(torch.float32),
                rtol=1e-1,
                atol=1e-1,
            )
        output = run_denox(
            "bench", reweighted_path,
            "--spec", "H=512", "W=512",
            verbose=True,
        )

        

