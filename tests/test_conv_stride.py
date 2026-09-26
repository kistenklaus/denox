from helpers import run_denox
import torch
import torch.nn as nn


class StridedConv(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv0 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv1 = nn.Conv2d(16, 3, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv1(torch.relu(self.conv0(x)))


def test_strided_conv_rejected(tmp_path):
    onnx_path = tmp_path / "net.onnx"

    dyn = torch.export.Dim.DYNAMIC
    with torch.no_grad():
        onnx = torch.onnx.export(
            StridedConv().half().eval(),
            (torch.ones(1, 3, 64, 64, dtype=torch.float16),),
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes={"x": {2: dyn, 3: dyn}},
            dynamo=True,
        )
    onnx.save(onnx_path)
    output = run_denox("compile", onnx_path, "-o", tmp_path / "net.dnx")
    assert output.returncode != 0
    assert "stride=(2,2)" in output.stdout
