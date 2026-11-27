import numpy as np
import onnx_ir
import pytest
import torch

from helpers import PROJECT_ROOT, run_denox


SELECT_ONNX = PROJECT_ROOT / "models" / "select.onnx"


@pytest.mark.parametrize(
    "flags",
    [
        pytest.param(["--fcoopmat=1"], id="coopmat"),
        pytest.param(["--fcoopmat=0"], id="no-coopmat"),
    ],
)
def test_select(tmp_path, flags):
    dnx_path = tmp_path / "select.dnx"
    input_path = tmp_path / "input.npy"
    output_path = tmp_path / "output.npy"

    program = torch.onnx.ONNXProgram(
        onnx_ir.load(SELECT_ONNX),
        exported_program=None,
    )
    input_tensor = torch.rand(
        (1, 8, 512, 512),
        dtype=torch.float16,
        generator=torch.Generator().manual_seed(0),
    )

    with torch.no_grad():
        output_ref = program(input_tensor)[0].to(torch.float32)

    result = run_denox(
        "compile",
        SELECT_ONNX,
        "-o",
        dnx_path,
        *flags,
    )
    assert result.returncode == 0, result.stdout

    # Denox's NPY convention is CHW, without the batch dimension.
    np.save(input_path, input_tensor[0].numpy())

    result = run_denox(
        "infer",
        dnx_path,
        "-i",
        input_path,
        "-o",
        output_path,
    )
    assert result.returncode == 0, result.stdout

    output = torch.from_numpy(np.load(output_path)).unsqueeze(0)

    assert output.shape == output_ref.shape
    torch.testing.assert_close(output, output_ref, rtol=0, atol=5e-2)
