import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import run_denox


class Pad(nn.Module):
    def __init__(self, padding, mode):
        super().__init__()
        self.padding = padding
        self.mode = mode

    def forward(self, x):
        return F.pad(x, self.padding, mode=self.mode)


@pytest.mark.parametrize(
    "channel,padding,mode",
    [
        # hwc
        pytest.param(10, (2, 2, 2, 2), "replicate", id="hwc-symmetric-replicate"),
        pytest.param(10, (2, 2, 2, 2), "constant", id="hwc-symmetric-constant"),

        pytest.param(10, (1, 2, 3, 4), "replicate", id="hwc8-asymmetric-replicate"),
        pytest.param(10, (1, 2, 3, 4), "constant", id="hwc8-asymmetric-constant"),

        #hwc8
        pytest.param(16, (2, 2, 2, 2), "replicate", id="hwc-symmetric-replicate"),
        pytest.param(16, (2, 2, 2, 2), "constant", id="hwc-symmetric-constant"),

        pytest.param(16, (1, 2, 3, 4), "replicate", id="hwc8-asymmetric-replicate"),
        pytest.param(16, (1, 2, 3, 4), "constant", id="hwc8-asymmetric-constant"),
    ],
)
def test_pad(tmp_path, channel, padding, mode):
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    input_path = tmp_path / "input.npy"
    output_path = tmp_path / "output.npy"

    model = Pad(padding, mode).eval()
    input_tensor = torch.rand(1, channel, 64, 64, dtype=torch.float16)

    dyn = torch.export.Dim.DYNAMIC
    with torch.no_grad():
        onnx_program = torch.onnx.export(
            model,
            (input_tensor,),
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes={"x": {2: dyn, 3: dyn}},
            dynamo=True,
        )
        output_ref = model(input_tensor).float()

    onnx_program.save(onnx_path)

    result = run_denox("compile", onnx_path, "-o", dnx_path)
    if result.returncode == 1 and "Failed to implement" in result.stderr:
        pytest.skip(f"Denox does not implement {mode} padding: {result.stderr}")
    assert result.returncode == 0, result.stderr

    # NPY input is CHW, without the batch dimension.
    np.save(input_path, input_tensor[0].numpy())

    result = run_denox("infer", dnx_path, "-i", input_path, "-o", output_path)
    assert result.returncode == 0, result.stderr

    output = torch.from_numpy(np.load(output_path)).unsqueeze(0)

    assert output.shape == output_ref.shape
    assert output.dtype == output_ref.dtype
    torch.testing.assert_close(output, output_ref, rtol=0, atol=0.00)
    

