from helpers import run_denox
from images import create_random_png
from inference import load_image, load_output_image
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class AvgPool(nn.Module):
    def __init__(self, kernel, channels):
        super().__init__()
        torch.manual_seed(0)
        self.kernel = kernel
        self.pre = nn.Conv2d(3, channels, 3, padding=1) if channels else None
        self.post = nn.Conv2d(channels, 3, 1) if channels else None
        if channels:
            with torch.no_grad():
                self.post.weight.mul_(0.5)
                self.post.bias.fill_(0.5)

    def forward(self, x):
        if self.pre is not None:
            x = self.pre(x)
        x = F.avg_pool2d(x, self.kernel)
        if self.post is not None:
            x = self.post(x)
        return x


@pytest.mark.parametrize("channels", [0, 12, 16])
@pytest.mark.parametrize("kernel, h, w", [
    (2, 12, 12), (3, 12, 12), (6, 12, 12), (12, 12, 12),
    (2, 37, 45), (3, 64, 64),
])
def test_avg_pool(cache_dir, tmp_path, kernel, channels, h, w):
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    db_path = cache_dir / "gpu.db"

    model = AvgPool(kernel, channels).eval()
    dyn = torch.export.Dim.DYNAMIC
    with torch.no_grad():
        onnx = torch.onnx.export(
            model.half(),
            (torch.ones(1, 3, 72, 72, dtype=torch.float16),),
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes={"x": {2: dyn, 3: dyn}},
            dynamo=True,
        )
    onnx.save(onnx_path)
    output = run_denox(
        "compile", onnx_path,
        f"--db={db_path}",
        "-o", dnx_path,
        "--min-samples=2",
        "--max-samples=2",
        "--relative-error=1",
    )
    assert output.returncode == 0
    create_random_png(input_path, w, h)
    output = run_denox("infer", dnx_path, "-i", input_path, "-o", output_path)
    assert output.returncode == 0

    x = load_image(input_path, 3).double()
    with torch.no_grad():
        ref = model.double()(x).clamp(0, 1)
    image = load_output_image(onnx, output_path)
    assert image.shape == ref.shape
    torch.testing.assert_close(image.double(), ref, rtol=0, atol=0.01)
