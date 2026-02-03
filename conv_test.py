import io
from typing import BinaryIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.dlpack

INPUT_CHANNELS_COUNT = 32
OUTPUT_CHANNEL_COUNT = 32


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            INPUT_CHANNELS_COUNT,
            OUTPUT_CHANNEL_COUNT,
            3,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dtype=torch.float16,
        )

    def forward(self, input):
        return self.conv(input)


net: nn.Module = Net()
net.to(dtype=torch.float16)

example_input = torch.ones(1, INPUT_CHANNELS_COUNT, 5, 5, dtype=torch.float16)
program = torch.onnx.export(
    net,
    (example_input,),
    dynamic_shapes={
        "input": {2: torch.export.Dim.DYNAMIC, 3: torch.export.Dim.DYNAMIC}
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
