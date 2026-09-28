from helpers import run_denox
from images import create_random_png
from inference import onnx_infer, load_output_image
import pytest
import torch
import torch.nn as nn


class ConcatNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv0 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv1 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, 3, padding=1)
        with torch.no_grad():
            self.conv2.bias.fill_(0.5)

    def forward(self, x):
        y = self.conv0(x)
        return self.conv2(torch.cat((self.conv1(y), y), 1))


@pytest.mark.parametrize("flags", [
    [], 
    ["--fmemcat=false"], 
    # to avoid concat+conv and make sure implicit-concat actually get's used.
    # --ffusion=false, doesn't disable memcat, but does disable concat+conv.
    ["--ffusion=false"], 
])
def test_implicit_concat(tmp_path, flags):
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"

    dyn = torch.export.Dim.DYNAMIC
    with torch.no_grad():
        onnx = torch.onnx.export(
            ConcatNet().half().eval(),
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
    torch.testing.assert_close(image.float(), ref.float(), rtol=0, atol=0.02)
