from helpers import run_denox
from models import SEGNET_MODELS
from inference import onnx_input_shape
import numpy as np
import pytest
import torch


@pytest.mark.parametrize("index", range(len(SEGNET_MODELS)))
def test_segnet(cache_dir, tmp_path, index):
    program = SEGNET_MODELS[index]
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    reweighted_path = tmp_path / "rnet.dnx"
    input_path = tmp_path / "input.npy"
    output_path = tmp_path / "output.npy"
    db_path = cache_dir / "gpu.db"

    program.save(onnx_path)
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
    output = run_denox("reweight", dnx_path, onnx_path, "-o", reweighted_path)
    assert output.returncode == 0

    _, c, h, w = onnx_input_shape(program)
    input = torch.rand((1, c, h, w), generator=torch.Generator().manual_seed(0))
    np.save(input_path, input[0].numpy().astype(np.float16))
    ref = program(input.to(torch.float16))[0][0].to(torch.float32)

    for dnx in (dnx_path, reweighted_path):
        output = run_denox("infer", dnx, "-i", input_path, "-o", output_path)
        assert output.returncode == 0
        image = torch.from_numpy(np.load(output_path))
        assert ref.shape == image.shape
        torch.testing.assert_close(image, ref, rtol=0, atol=5e-2)
