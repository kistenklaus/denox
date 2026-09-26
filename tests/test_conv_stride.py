from helpers import run_denox
from images import create_random_png
from inference import load_image, load_output_image
import pytest
import torch
import torch.nn as nn


class StridedConv(nn.Module):
    def __init__(self, cin, cout, kernel, stride, padding, relu):
        super().__init__()
        torch.manual_seed(0)
        self.pre = nn.Conv2d(3, cin, 3, padding=1) if cin != 3 else nn.Identity()
        self.conv = nn.Conv2d(cin, cout, kernel, stride=stride, padding=padding)
        self.post = nn.Conv2d(cout, 3, 1)
        self.relu = relu
        with torch.no_grad():
            self.post.weight.mul_(0.5)
            self.post.bias.fill_(0.5)

    def forward(self, x):
        x = self.conv(self.pre(x))
        if self.relu:
            x = torch.relu(x)
        return self.post(x)


class UpsampleStridedConv(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.pre = nn.Conv2d(3, 16, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.post = nn.Conv2d(32, 3, 1)
        with torch.no_grad():
            self.post.weight.mul_(0.5)
            self.post.bias.fill_(0.5)

    def forward(self, x):
        return self.post(self.conv(self.up(self.pre(x))))


CHANNELS = [(8, 24), (32, 64), (64, 128)]


@pytest.mark.parametrize("cin, cout, kernel, stride, padding, relu", [
    (3, 16, 3, 2, 1, True),
    *[(cin, cout, 3, 2, 1, True) for cin, cout in CHANNELS],
    *[(cin, cout, 1, 2, 0, False) for cin, cout in CHANNELS],
])
@pytest.mark.parametrize("h, w", [(64, 64), (37, 45)])
def test_conv_stride(tmp_path, cin, cout, kernel, stride, padding, relu, h, w):
    model = StridedConv(cin, cout, kernel, stride, padding, relu)
    run_strided_conv(tmp_path, model, h, w)


@pytest.mark.parametrize("kernel, stride, padding", [
    (3, 3, 1),
    (3, (2, 1), 1),
    (5, 2, 2),
])
def test_conv_stride_other(tmp_path, kernel, stride, padding):
    model = StridedConv(16, 32, kernel, stride, padding, False)
    run_strided_conv(tmp_path, model, 41, 50)


@pytest.mark.parametrize("flags", [[], ["--ffusion=false"]])
def test_conv_stride_upsample(tmp_path, flags):
    run_strided_conv(tmp_path, UpsampleStridedConv(), 37, 45, flags)
    output = run_denox("dump", tmp_path / "net.dnx")
    assert output.returncode == 0
    assert ("basic-upsample" in output.stdout) == bool(flags)


def run_strided_conv(tmp_path, model, h, w, flags=()):
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"

    model.eval()
    dyn = torch.export.Dim.DYNAMIC
    with torch.no_grad():
        onnx = torch.onnx.export(
            model.half(),
            (torch.ones(1, 3, 64, 64, dtype=torch.float16),),
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes={"x": {2: dyn, 3: dyn}},
            dynamo=True,
        )
    onnx.save(onnx_path)
    output = run_denox("compile", onnx_path, "-o", dnx_path, *flags)
    assert output.returncode == 0
    create_random_png(input_path, w, h)
    output = run_denox("infer", dnx_path, "-i", input_path, "-o", output_path)
    assert output.returncode == 0

    x = load_image(input_path, 3).double()
    with torch.no_grad():
        ref = model.double()(x).clamp(0, 1)
    image = load_output_image(onnx, output_path)
    assert image.shape == ref.shape
    torch.testing.assert_close(image.double(), ref, rtol=0, atol=0.02)
