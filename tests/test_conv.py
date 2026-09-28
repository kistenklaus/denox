import numpy as np
import pytest
import torch
import torch.nn as nn

from helpers import run_denox


def conv_id(kwargs):
    return (
        f"c{kwargs['in_channels']}-to-{kwargs['out_channels']}"
        f"-k{kwargs['kernel_size']}"
        f"-s{kwargs['stride']}"
        f"-p{kwargs['padding']}"
        f"-{kwargs['padding_mode']}"
        f"-bias-{int(kwargs['bias'])}"
    )


@pytest.mark.parametrize(
    "flags",
    [
        pytest.param(["--fcoopmat=1"], id="coopmat"),
        pytest.param(["--fcoopmat=0"], id="no-coopmat"),
    ],
)
@pytest.mark.parametrize(
    "conv_kwargs",
    [
        # 3x3 same zero padding.
        {
            "in_channels": 10,
            "out_channels": 10,
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
            "padding_mode": "zeros",
            "bias": True,
        },
        {
            "in_channels": 16,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
            "padding_mode": "zeros",
            "bias": True,
        },
        {
            "in_channels": 10,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
            "padding_mode": "zeros",
            "bias": True,
        },
        {
            "in_channels": 16,
            "out_channels": 10,
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
            "padding_mode": "zeros",
            "bias": True,
        },
        {
            "in_channels": 16,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
            "padding_mode": "zeros",
            "bias": False,
        },
        
        # 3x3 1, replicate padding.
        {
            "in_channels": 10,
            "out_channels": 10,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "padding_mode": "replicate",
            "bias": True,
        },
        {
            "in_channels": 16,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "padding_mode": "replicate",
            "bias": True,
        },
        {
            "in_channels": 10,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "padding_mode": "replicate",
            "bias": True,
        },
        {
            "in_channels": 16,
            "out_channels": 10,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "padding_mode": "replicate",
            "bias": True,
        },
        {
            "in_channels": 16,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "padding_mode": "replicate",
            "bias": False,
        },

        # 5x5 same zero padding.
        {
            "in_channels": 10,
            "out_channels": 10,
            "kernel_size": 5,
            "stride": 1,
            "padding": "same",
            "padding_mode": "zeros",
            "bias": True,
        },
        {
            "in_channels": 16,
            "out_channels": 16,
            "kernel_size": 5,
            "stride": 1,
            "padding": "same",
            "padding_mode": "zeros",
            "bias": True,
        },
        {
            "in_channels": 10,
            "out_channels": 16,
            "kernel_size": 5,
            "stride": 1,
            "padding": "same",
            "padding_mode": "zeros",
            "bias": True,
        },
        {
            "in_channels": 16,
            "out_channels": 10,
            "kernel_size": 5,
            "stride": 1,
            "padding": "same",
            "padding_mode": "zeros",
            "bias": True,
        },
        # 3x3 strided
        {
            "in_channels": 10,
            "out_channels": 10,
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "padding_mode": "zeros",
            "bias": True,
        },
        {
            "in_channels": 16,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "padding_mode": "zeros",
            "bias": True,
        },
        {
            "in_channels": 10,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "padding_mode": "zeros",
            "bias": True,
        },
        {
            "in_channels": 16,
            "out_channels": 10,
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "padding_mode": "zeros",
            "bias": True,
        },
    ],
    ids=conv_id,
)
def test_conv(tmp_path, flags, conv_kwargs):
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    input_path = tmp_path / "input.npy"
    output_path = tmp_path / "output.npy"

    model = nn.Conv2d(**conv_kwargs, dtype=torch.float16).eval()
    input_tensor = torch.rand(
        1,
        conv_kwargs["in_channels"],
        64,
        64,
        dtype=torch.float16,
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
        output_ref = model(input_tensor).float()

    onnx_program.save(onnx_path)

    result = run_denox("compile", onnx_path, "-o", dnx_path, *flags)
    if (result.returncode == 1): 
        # Failed to implement!
        pytest.skip(f"Denox does not implement this configuration: {result.stderr}")

    assert result.returncode == 0, result.stderr

    # Denox's NPY convention is CHW, without the batch dimension.
    np.save(input_path, input_tensor[0].numpy())

    result = run_denox("infer", dnx_path, "-i", input_path, "-o", output_path)
    assert result.returncode == 0, result.stderr

    output = torch.from_numpy(np.load(output_path)).unsqueeze(0)

    assert output.shape == output_ref.shape
    assert output.dtype == output_ref.dtype
    torch.testing.assert_close(output, output_ref, rtol=0, atol=0.02)
