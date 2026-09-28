from helpers import run_denox
from images import create_random_png
from inference import onnx_infer, load_output_image
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class PaddedConv(nn.Module):
    def __init__(self, **conv1_args):
        super().__init__()
        torch.manual_seed(0)
        self.conv0 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv1 = nn.Conv2d(8, 3, 3, **conv1_args)
        with torch.no_grad():
            self.conv1.bias.fill_(0.5)

    def forward(self, x):
        return self.conv1(F.relu(self.conv0(x)))


@pytest.mark.parametrize("conv1_args, flags", [
    # exports as Pad(mode=edge) + Conv(pads=0)
    (dict(padding=1, padding_mode="replicate"), []),
    (dict(padding=1, padding_mode="replicate"), ["--ffusion=false"]),
    (dict(padding=0), []),
])
def test_conv_padding(tmp_path, conv1_args, flags):
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"

    dyn = torch.export.Dim.DYNAMIC
    with torch.no_grad():
        onnx = torch.onnx.export(
            PaddedConv(**conv1_args).half().eval(),
            (torch.ones(1, 3, 64, 64, dtype=torch.float16),),
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes={"x": {2: dyn, 3: dyn}},
            dynamo=True,
        )
    onnx.save(onnx_path)
    output = run_denox("compile", onnx_path, "-o", dnx_path, *flags)
    assert output.returncode == 0
    create_random_png(input_path, 64, 64)
    output = run_denox("infer", dnx_path, "-i", input_path, "-o", output_path)
    assert output.returncode == 0

    ref = onnx_infer(onnx, input_path).clamp(0, 1)
    image = load_output_image(onnx, output_path)
    assert image.shape == ref.shape
    torch.testing.assert_close(image.float(), ref.float(), rtol=0, atol=0.02)
