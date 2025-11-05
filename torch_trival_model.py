import io
from typing import BinaryIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.dlpack
from denox import DataType, Layout, Module, Shape, Storage, TargetEnv

INPUT_CHANNELS_COUNT = 3
OUTPUT_CHANNEL_COUNT = 16


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # pm = "zeros"
        # self.enc0 = nn.Conv2d(
        #     INPUT_CHANNELS_COUNT, OUTPUT_CHANNEL_COUNT, 3, padding="same", padding_mode=pm, bias=False
        # )
        #
        # self.con0 = nn.Conv2d(112, 112, 3, padding="same", padding_mode=pm)
        # self.dec0 = nn.Conv2d(32, 16, 3, padding="same", padding_mode=pm)
        #
        # self.pool = nn.MaxPool2d(2, 2)
        # self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        # H, W = I.size(2), I.size(3)
        # alignment = 2  # ensure even H/W so pool+upsample align perfectly
        # H, W = I.size(2), I.size(3)
        # pad_w = (alignment - (W % alignment)) % alignment
        # pad_h = (alignment - (H % alignment)) % alignment
        # I_aligned = F.pad(I, (0, pad_w, 0, pad_h), mode="replicate")
        # extr = self.enc0(I_aligned)
        # x_128 = self.pool(extr);
        # x = self.enc0(I)

        x = x[:,:,1:3,1:3]
        return x


example_input = torch.ones(1, INPUT_CHANNELS_COUNT, 100, 100, dtype=torch.float16)

net : nn.Module = Net()

# weight = torch.ones((OUTPUT_CHANNEL_COUNT, INPUT_CHANNELS_COUNT, 3, 3), dtype=torch.float16)
# with torch.no_grad():
#     net.enc0.weight.copy_(weight)

program = torch.onnx.export(
    net,
    (example_input,),
    dynamic_shapes={"x": {2: torch.export.Dim.DYNAMIC, 3: torch.export.Dim.DYNAMIC}},
    input_names=["input"],
    output_names=["output"]
)
program.save("net.onnx")
# dnx = Module.compile(
#     program,
#     input_shape=Shape(H="H", W="W"),
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
