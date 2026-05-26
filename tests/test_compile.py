from helpers import run_denox
from models import MODELS;
from images import create_random_png
from inference import onnx_infer, load_output_image
import torch

def test_compile(cache_dir, tmp_path):
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    input_path = tmp_path / "input.png"
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
            "-o", output_path,
            verbose=True,
        )
        assert output.returncode == 0
        ref = onnx_infer(onnx, input_path)
        if ref is not None:
            image = load_output_image(onnx, output_path)
            assert image is not None
            assert ref.shape == image.shape
            torch.testing.assert_close(
                image.to(torch.float32),
                ref.to(torch.float32),
                rtol=1e-1,
                atol=1e-1,
            )


def test_compile_descriptor_policies(cache_dir, tmp_path):
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    db_path = cache_dir / "gpu.db"

    for onnx in MODELS:
        onnx.save(onnx_path)
        output = run_denox(
            "compile", onnx_path,
            f"--db={db_path}",
            "-o", dnx_path,
            "--use-descriptor-sets", "1", "1", "1", "1", "1",
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
            "-o", output_path,
            verbose=True,
        )
        assert output.returncode == 0
        ref = onnx_infer(onnx, input_path)
        if ref is not None:
            image = load_output_image(onnx, output_path)
            assert image is not None
            assert ref.shape == image.shape
            torch.testing.assert_close(
                image.to(torch.float32),
                ref.to(torch.float32),
                rtol=1e-1,
                atol=1e-1,
            )


def test_compile_shape_format_type(cache_dir, tmp_path):
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    db_path = cache_dir / "gpu.db"

    for onnx in MODELS:
        onnx.save(onnx_path)
        output = run_denox(
            "compile", onnx_path,
            f"--db={db_path}",
            "-o", dnx_path,
            "--shape", "input=H:W:C", "output=OH:OW:OC",
            "--storage", "input=ssbo", "output=ssbo",
            "--format", "input=hwc", "output=hwc",
            "--type", "input=f16", "output=f16",
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
            "-o", output_path,
            verbose=True,
        )
        assert output.returncode == 0
        ref = onnx_infer(onnx, input_path)
        if ref is not None:
            image = load_output_image(onnx, output_path)
            assert image is not None
            assert ref.shape == image.shape
            torch.testing.assert_close(
                image.to(torch.float32),
                ref.to(torch.float32),
                rtol=1e-1,
                atol=1e-1,
            )

def test_compile_nofeatures(cache_dir, tmp_path):
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    db_path = cache_dir / "gpu.db"

    for onnx in MODELS:
        onnx.save(onnx_path)
        output = run_denox(
            "compile", onnx_path,
            f"--db={db_path}",
            "-o", dnx_path,
            "--fcoopmat=0",
            "--ffusion=0",
            "--fmemcat=0",
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
            "-o", output_path,
            verbose=True,
        )
        assert output.returncode == 0
        ref = onnx_infer(onnx, input_path)
        if ref is not None:
            image = load_output_image(onnx, output_path)
            assert image is not None
            assert ref.shape == image.shape
            torch.testing.assert_close(
                image.to(torch.float32),
                ref.to(torch.float32),
                rtol=1e-1,
                atol=1e-1,
            )

def test_compile_quite(cache_dir, tmp_path):
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    input_path = tmp_path / "input.png"
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
            "-q",
            verbose=True,
        )
        assert output.returncode == 0
        output = output.stderr + output.stdout
        assert not output.strip()

# def test_compile_with_device_yml(cache_dir, tmp_path):
#     onnx_path = tmp_path / "net.onnx"
#     dnx_path = tmp_path / "net.dnx"
#     input_path = tmp_path / "input.png"
#     output_path = tmp_path / "output.png"
#     db_path = cache_dir / "gpu.db"
#     device_yml = tmp_path / "gpu.yaml";
#
#     output = run_denox("query-device-info", "-o", device_yml);
#     assert output.returncode == 0
#
#     for onnx in MODELS:
#         onnx.save(onnx_path)
#         output = run_denox(
#             "compile", onnx_path,
#             f"--db={db_path}",
#             "-o", dnx_path,
#             "--min-samples=2",
#             "--max-samples=2",
#             "--relative-error=1",
#             "--device", device_yml,
#             verbose=True,
#         )
#         assert output.returncode == 0
#         create_random_png(input_path, 1920, 1080)
#         output = run_denox(
#             "infer", dnx_path,
#             "-i", input_path,
#             "-o", output_path,
#             verbose=True,
#         )
#         assert output.returncode == 0
#         ref = onnx_infer(onnx, input_path)
#         if ref is not None:
#             image = load_output_image(onnx, output_path)
#             assert image is not None
#             assert ref.shape == image.shape
#             torch.testing.assert_close(
#                 image.to(torch.float32),
#                 ref.to(torch.float32),
#                 rtol=1e-1,
#                 atol=1e-1,
#             )
