from numpy import shape
import torch
import torch.nn as nn
import torch.utils.dlpack

from denox import Layout, Module, Shape


class WrapperModel(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.op = x

    def forward(self, x):
        return self.op(x)


def tensor_error_report(ref: torch.Tensor,
                        test: torch.Tensor,
                        name: str = "output",
                        eps: float = 1e-12,
                        dynamic_max: float | None = None):
    """
    Prints a concise summary of absolute + relative errors, percentiles,
    cosine similarity, and PSNR. Works with any shape/dtype/device.
    """
    ref32  = ref.detach().to(torch.float32)
    test32 = test.detach().to(torch.float32)
    diff   = (test32 - ref32)

    # Absolute errors
    abs_diff = diff.abs()
    abs_max  = abs_diff.max().item()
    mae      = abs_diff.mean().item()
    rmse     = (abs_diff.pow(2).mean().sqrt()).item()
    p90_abs  = torch.quantile(abs_diff.flatten(), 0.90).item()
    p99_abs  = torch.quantile(abs_diff.flatten(), 0.99).item()
    p999_abs = torch.quantile(abs_diff.flatten(), 0.999).item()

    # Brief, readable printout
    print(f"\n=== {name} error report ===")
    print(f"shape={tuple(ref.shape)} dtype_ref={ref.dtype} dtype_test={test.dtype} device_ref={ref.device} device_test={test.device}")
    print(f"Error:   max={abs_max:.6g}   p90={p90_abs:.6g}    p99={p99_abs:.6g}   p99.9={p999_abs:.6g}   mean={mae:.6g}   rmse={rmse:.6g}")

def run_module_test(
    module: nn.Module,
    input: torch.Tensor,
    rtol=1e-3,
    atol=1e-2,
    input_layout=Layout.Undefined,
    output_layout=Layout.Undefined,
):
    net = WrapperModel(module)
    program = torch.onnx.export(
        net,
        (input,),
        dynamo=True,
        export_params=True,
        external_data=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={
            "x": {2: torch.export.Dim.DYNAMIC, 3: torch.export.Dim.DYNAMIC}
        },
    )
    dnx = Module.compile(
        program,
        input_shape=Shape(H="H", W="W"),
        input_layout=input_layout,
        output_layout=output_layout,
        quiet=True,
        # verbose=True,
        # summary=True,
    )
    output = torch.utils.dlpack.from_dlpack(dnx(input))
    eval_model = net.eval()
    device = torch.cpu.current_device()
    if (torch.cuda.is_available()):
        device = torch.cuda.current_device()

    eval_model = eval_model.to(device)
    input = input.to(device)
    output = output.to(device)
    expected = net(input)

    assert output.size() == expected.size()

    # print("OUTPUT:")
    # print(output)
    # print("EXPECTED:")
    # print(expected)
    # #
    # print("DIFFERENCE:")
    # print(output - expected)
    # print(torch.max(torch.abs(output - expected)))

    if not torch.allclose(output, expected, rtol=rtol, atol=atol):
        print(output)
        tensor_error_report(output, expected)
        assert False
