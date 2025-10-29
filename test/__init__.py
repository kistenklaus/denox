import torch
import torch.nn as nn
import torch.utils.dlpack

from denox import Module, Shape


class WrapperModel(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.op = x

    def forward(self, x):
        return self.op(x)


def run_module_test(module: nn.Module, input: torch.Tensor, rtol=1e-2, atol=1e-3):
    net = WrapperModel(module)
    program = torch.onnx.export(
        net,
        (input,),
        dynamic_shapes={
            "x": {2: torch.export.Dim.DYNAMIC, 3: torch.export.Dim.DYNAMIC}
        },
    )
    dnx = Module.compile(
        program,
        input_shape=Shape(H="H", W="W"),
        quiet=True,
        # verbose=True,
        # summary=True,
    )
    output = torch.utils.dlpack.from_dlpack(dnx(input))
    expected = net(input)

    print(output)
    print(expected)
    
    print(output - expected)
    print(torch.max(output - expected))
    assert torch.allclose(output, expected, rtol=rtol, atol=atol)
