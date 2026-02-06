import io
from typing import BinaryIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.dlpack

A_CHAN = 96
B_CHAN = 64
OUT_CHAN = 64
OUTPUT_CHANNEL_COUNT = 8


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        

        self.convb = nn.Conv2d(
            A_CHAN,
            B_CHAN,
            3,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dtype=torch.float16,
        )

        self.convab = nn.Conv2d(
            A_CHAN + B_CHAN,
            OUT_CHAN,
            3,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dtype=torch.float16,
        )

        self.convout = nn.Conv2d(
            OUT_CHAN,
            OUTPUT_CHANNEL_COUNT,
            3,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dtype=torch.float16,
        )

    def forward(self, a):
        b = self.convb(a)
        ab = torch.cat([a,b], 1)
        out = self.convab(ab)
        return self.convout(out)

net: nn.Module = Net()
net.to(dtype=torch.float16)

example_input = torch.ones(1, A_CHAN, 5, 5, dtype=torch.float16)
program = torch.onnx.export(
    net,
    (example_input,),
    dynamic_shapes={
        "a": {2: torch.export.Dim.DYNAMIC, 3: torch.export.Dim.DYNAMIC}
    },
    input_names=["input"],
    output_names=["output"],
)
program.save("net.onnx")

# output = net(example_input)
#
# print(output)
#
# dnx = denox.Module.compile(
#     program,
#     input_shape=denox.Shape(H="H", W="W"),
#     summary=True,
#     verbose=True,
# )
#
# dnx.save("net.dnx")

# dreams:
# output = torch.utils.dlpack.from_dlpack(dnx(example_input))
# expected = net(example_input)
# print(output)
# print(expected)
