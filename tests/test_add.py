from helpers import run_denox
from images import create_random_png
from inference import load_image, load_output_image
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class Add(nn.Module):
    def __init__(self, channels, activation):
        super().__init__()
        self.a = nn.Conv2d(3, channels, 3, padding=1)
        self.b = nn.Conv2d(3, channels, 3, padding=1)
        self.post = nn.Conv2d(channels, 3, 1)
        self.activation = activation

    def forward(self, x):
        y = self.a(x) + self.b(x)
        if self.activation == "relu":
            y = F.relu(y)
        elif self.activation == "leaky_relu":
            y = F.leaky_relu(y, 0.1)
        return self.post(y)


class AddInput(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(3, 3, 3, padding=1)
        with torch.no_grad():
            self.conv.weight.mul_(0.25)

    def forward(self, x):
        return x + self.conv(x)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        if cin != cout:
            self.shortcut = nn.Conv2d(cin, cout, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        y = self.conv2(F.relu(self.conv1(x)))
        return F.relu(y + self.shortcut(x))


class Residual(nn.Module):
    def __init__(self, cout):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.block0 = BasicBlock(16, 16)
        self.block1 = BasicBlock(16, cout)
        self.post = nn.Conv2d(cout, 3, 1)

    def forward(self, x):
        x = F.relu(self.stem(x))
        return self.post(self.block1(self.block0(x)))


def init_weights(model):
    torch.manual_seed(0)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=(2 / m.weight[0].numel()) ** 0.5)
            nn.init.uniform_(m.bias, -0.1, 0.1)
    with torch.no_grad():
        model.post.weight.mul_(0.25)
        model.post.bias.fill_(0.5)
    return model


def run_model(tmp_path, model, h, w, flags=()):
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


@pytest.mark.parametrize("channels", [12, 16])
@pytest.mark.parametrize("activation", [None, "relu", "leaky_relu"])
@pytest.mark.parametrize("h, w", [(64, 64), (37, 45)])
def test_add(tmp_path, channels, activation, h, w):
    run_model(tmp_path, init_weights(Add(channels, activation)), h, w)
    output = run_denox("dump", tmp_path / "net.dnx")
    assert output.returncode == 0
    assert "basic-activation" not in output.stdout


def test_add_input(tmp_path):
    run_model(tmp_path, AddInput(), 37, 45)


@pytest.mark.parametrize("flags", [[], ["--ffusion=false"]])
@pytest.mark.parametrize("h, w", [(64, 64), (38, 46)])
def test_residual_identity(tmp_path, flags, h, w):
    run_model(tmp_path, init_weights(Residual(16)), h, w, flags)


@pytest.mark.parametrize("h, w", [(64, 64), (38, 46)])
def test_residual_projection(tmp_path, h, w):
    run_model(tmp_path, init_weights(Residual(32)), h, w)
