from helpers import run_denox
from models import export_model, YetAnotherUNet
from images import create_random_png
from inference import onnx_infer, load_output_image
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, padding="same",
                     padding_mode="zeros", dtype=torch.float16)


class InnerUNet(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, depth):
        super().__init__()
        self.conv_in = Conv(in_channels, out_channels)
        self.enc = nn.ModuleList([
            Conv(out_channels if i == 0 else mid_channels, mid_channels)
            for i in range(depth)
        ])
        self.bottom = Conv(mid_channels, mid_channels)
        self.dec = nn.ModuleList([
            Conv(2 * mid_channels, out_channels if i == 0 else mid_channels)
            for i in range(depth)
        ])

    def forward(self, x):
        x = F.relu(self.conv_in(x))
        skips = []
        for i, enc in enumerate(self.enc):
            if i > 0:
                x = F.max_pool2d(x, 2, 2)
            x = F.relu(enc(x))
            skips.append(x)
        x = F.relu(self.bottom(x))
        for i in reversed(range(len(self.dec))):
            x = F.relu(self.dec[i](torch.cat((x, skips[i]), 1)))
            if i > 0:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x


# U-Net of U-Nets (like U^2-Net), the schedule search explodes on these.
class NestedUNet(nn.Module):
    def __init__(self, outer_depth=2, inner_depth=2, ch=8):
        super().__init__()
        self.input_channels = 3
        self.alignment = 2 ** (outer_depth + inner_depth - 2)
        self.enc = nn.ModuleList([
            InnerUNet(3 if i == 0 else ch, ch, ch, inner_depth)
            for i in range(outer_depth)
        ])
        self.dec = nn.ModuleList([
            InnerUNet(2 * ch, ch, ch, inner_depth)
            for _ in range(outer_depth - 1)
        ])
        self.conv_out = Conv(ch, 3)

    def forward(self, input):
        skips = []
        x = input
        for i, enc in enumerate(self.enc):
            if i > 0:
                x = F.max_pool2d(x, 2, 2)
            x = enc(x)
            skips.append(x)
        for i in reversed(range(len(self.dec))):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self.dec[i](torch.cat((x, skips[i]), 1))
        return F.relu(self.conv_out(x))


def compile_model(cache_dir, onnx_path, dnx_path, *flags):
    output = run_denox(
        "compile", onnx_path,
        f"--db={cache_dir / 'gpu.db'}",
        "-o", dnx_path,
        *flags,
        "--min-samples=2",
        "--max-samples=2",
        "--relative-error=1",
        timeout=600,
        verbose=True,
    )
    assert output.returncode == 0
    return output


def check_infer(onnx, dnx_path, tmp_path, atol=1e-2):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    create_random_png(input_path, 1920, 1080)
    output = run_denox(
        "infer", dnx_path,
        "-i", input_path,
        "-o", output_path,
        verbose=True,
    )
    assert output.returncode == 0
    ref = onnx_infer(onnx, input_path)
    image = load_output_image(onnx, output_path)
    assert ref.shape == image.shape
    torch.testing.assert_close(
        image.to(torch.float32),
        ref.to(torch.float32),
        rtol=1e-1,
        atol=atol,
    )


def test_compile_max_search_states(cache_dir, tmp_path):
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    torch.manual_seed(0)
    onnx = export_model(NestedUNet())
    onnx.save(onnx_path)
    output = compile_model(cache_dir, onnx_path, dnx_path,
                           "--max-search-states", "0")
    assert "exceeded 0 states" in output.stdout
    check_infer(onnx, dnx_path, tmp_path)


# YetAnotherUNet fits the default budget, its minimum-dispatch schedule is not
# the greedy one
@pytest.mark.parametrize("model", [NestedUNet, YetAnotherUNet])
def test_reweight_max_search_states(cache_dir, tmp_path, model):
    onnx_path = tmp_path / "net.onnx"
    new_onnx_path = tmp_path / "new_net.onnx"
    dnx_path = tmp_path / "net.dnx"
    reweighted_path = tmp_path / "rnet.dnx"
    torch.manual_seed(0)
    export_model(model()).save(onnx_path)
    output = compile_model(cache_dir, onnx_path, dnx_path,
                           "--max-search-states", "0")
    assert "exceeded 0 states" in output.stdout

    torch.manual_seed(1)
    new_onnx = export_model(model())
    new_onnx.save(new_onnx_path)
    output = run_denox(
        "reweight", dnx_path, new_onnx_path,
        "-o", reweighted_path,
        timeout=600,
        verbose=True,
    )
    assert output.returncode == 0
    check_infer(new_onnx, reweighted_path, tmp_path, atol=1e-1)
