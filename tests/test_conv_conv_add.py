import numpy as np
import pytest
import torch
import torch.nn as nn

from helpers import run_denox


class ConvConvAdd(nn.Module):
    def __init__(self, conv_a_kwargs, conv_b_kwargs, relu, upsample_mode):
        super().__init__()
        self.helper = nn.Conv2d(
            conv_a_kwargs["in_channels"],
            conv_b_kwargs["in_channels"],
            1,
            1,
            padding="same",
            bias=conv_a_kwargs["bias"] and conv_b_kwargs["bias"],
        )
        self.conv0 = nn.Conv2d(**conv_a_kwargs)
        self.conv1 = nn.Conv2d(**conv_b_kwargs)
        self.relu = relu
        if upsample_mode == "bilinear":
            self.upsample = nn.Upsample(
                scale_factor=2, mode=upsample_mode, align_corners=False
            )
        elif upsample_mode == "nearest":
            self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
        else:
            self.upsample = None

    def forward(self, input):
        if self.upsample is not None:
            input = self.upsample(input)
        a = input
        b = self.helper(input)
        output = self.conv0(a) + self.conv1(b)
        return torch.relu(output) if self.relu else output


@pytest.mark.parametrize(
    "flags",
    [
        pytest.param(["--fcoopmat=1"], id="coopmat"),
    ],
)
@pytest.mark.parametrize(
    "conv_a_kwargs, conv_b_kwargs, relu, upsample_mode, input_size",
    [
        pytest.param(
            {
                "in_channels": 8,
                "out_channels": 8,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "padding_mode": "zeros",
                "bias": False,
            },
            {
                "in_channels": 8,
                "out_channels": 8,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "padding_mode": "zeros",
                "bias": False,
            },
            False,
            None,
            512,
            id="8ch-3x3-s1-p1-no-bias",
        ),
        pytest.param(
            {
                "in_channels": 8,
                "out_channels": 8,
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
                "padding_mode": "zeros",
                "bias": True,
            },
            {
                "in_channels": 8,
                "out_channels": 8,
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
                "padding_mode": "zeros",
                "bias": True,
            },
            True,
            None,
            512,
            id="8ch-1x1-s1-p0-bias",
        ),
        pytest.param(
            {
                "in_channels": 8,
                "out_channels": 8,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "padding_mode": "zeros",
                "bias": True,
            },
            {
                "in_channels": 8,
                "out_channels": 8,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "padding_mode": "zeros",
                "bias": True,
            },
            True,
            None,
            512,
            id="8ch-3x3-s2-p1-bias-relu",
        ),
        pytest.param(
            {
                "in_channels": 7,
                "out_channels": 11,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "padding_mode": "zeros",
                "bias": False,
            },
            {
                "in_channels": 13,
                "out_channels": 11,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "padding_mode": "zeros",
                "bias": True,
            },
            True,
            None,
            512,
            id="7x13-to-11ch-3x3-relu",
        ),
        # pytest.param(
        #     {
        #         "in_channels": 512,
        #         "out_channels": 512,
        #         "kernel_size": 3,
        #         "stride": 1,
        #         "padding": 1,
        #         "padding_mode": "zeros",
        #         "bias": False,
        #     },
        #     {
        #         "in_channels": 256,
        #         "out_channels": 512,
        #         "kernel_size": 3,
        #         "stride": 1,
        #         "padding": 1,
        #         "padding_mode": "zeros",
        #         "bias": True,
        #     },
        #     False,
        #     None,
        #     512,
        #     id="512x256-to-512ch-3x3",
        # ),
        pytest.param(
            {
                "in_channels": 8,
                "out_channels": 8,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "padding_mode": "zeros",
                "bias": False,
            },
            {
                "in_channels": 8,
                "out_channels": 8,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "padding_mode": "zeros",
                "bias": False,
            },
            False,
            "nearest",
            64,
            id="8ch-3x3-nearest",
        ),
        pytest.param(
            {
                "in_channels": 8,
                "out_channels": 8,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "padding_mode": "zeros",
                "bias": False,
            },
            {
                "in_channels": 8,
                "out_channels": 8,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "padding_mode": "zeros",
                "bias": False,
            },
            True,
            "bilinear",
            64,
            id="8ch-3x3-bilinear-relu",
        ),
    ],
)
def test_conv_conv_add(
    tmp_path,
    flags,
    conv_a_kwargs,
    conv_b_kwargs,
    relu,
    upsample_mode,
    input_size,
):

    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    input_path = tmp_path / "input.npy"
    output_path = tmp_path / "output.npy"

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        ConvConvAdd(conv_a_kwargs, conv_b_kwargs, relu, upsample_mode)
        .half()
        .eval()
        .to(device)
    )
    input_tensor = torch.rand(
        1,
        conv_a_kwargs["in_channels"],
        input_size,
        input_size,
        dtype=torch.float16,
        device=device,
    )

    with torch.no_grad():
        onnx_program = torch.onnx.export(
            model,
            (input_tensor,),
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes=(
                {
                    2: torch.export.Dim("height"),
                    3: torch.export.Dim("width"),
                },
            ),
            dynamo=True,
        )
        output_ref = model(input_tensor).float().cpu()

    onnx_program.save(onnx_path)

    result = run_denox("compile", onnx_path, "-o", dnx_path, *flags)
    if result.returncode == 1:
        pytest.skip(result.stderr)
    assert result.returncode == 0, result.stderr

    np.save(input_path, input_tensor[0].cpu().numpy())
    result = run_denox("infer", dnx_path, "-i", input_path, "-o", output_path)
    assert result.returncode == 0, result.stderr

    output = torch.from_numpy(np.load(output_path)).unsqueeze(0)
    assert output.shape == output_ref.shape
    assert output.dtype == output_ref.dtype
    torch.testing.assert_close(output, output_ref, rtol=0, atol=0.02)
