# tza.py
from __future__ import annotations
import struct
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import torch
import time

import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image

INPUT_CHANNELS_COUNT = 3
OUTPUT_CHANNEL_COUNT = 3


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        ch_tmp = 128
        ch_a = 64
        ch_b = 32
        ch_out = 64

        self.conv0 = nn.Conv2d(
            INPUT_CHANNELS_COUNT,
            ch_b,
            3,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dtype=torch.float16,
        )
        self.conv00 = nn.Conv2d(
            ch_b,
            ch_tmp,
            3,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dtype=torch.float16,
        )
        self.convx = nn.Conv2d(
            ch_tmp,
            ch_tmp,
            3,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dtype=torch.float16,
        )
        self.conva = nn.Conv2d(
            ch_tmp,
            ch_a,
            3,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dtype=torch.float16,
        )

        self.conv1 = nn.Conv2d(
            ch_a + ch_b,
            ch_out,
            3,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dtype=torch.float16,
        )

        self.conv2 = nn.Conv2d(
            ch_out,
            OUTPUT_CHANNEL_COUNT,
            3,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dtype=torch.float16,
        )
    def forward(self, input):
        b = self.conv0(input)
        a = self.conv00(b)
        a = self.convx(a)
        a = self.convx(a)
        a = self.convx(a)
        a = self.convx(a)
        a = self.convx(a)
        a = self.conva(a)

        ab = torch.cat([a,b], 1)
        o = self.conv1(ab)


        o = self.conv2(o)
        return o


net: nn.Module = Net()
net.to(dtype=torch.float16)

example_input = torch.ones(1, INPUT_CHANNELS_COUNT, 1080, 1920, dtype=torch.float16)
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

img = Image.open("input.png").convert("RGB")

to_tensor = transforms.ToTensor()
input_tensor: torch.Tensor = to_tensor(img).unsqueeze(0).to(dtype=torch.float16)

# output_tensor = torch.utils.dlpack.from_dlpack(dnx(input_tensor))
# output_tensor = output_tensor.squeeze(0)
# output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
#
rt_ldr = net.eval()
device = torch.cuda.current_device()
rt_ldr = rt_ldr.to(device=device)
input_tensor = input_tensor.to(device=device)
#
output_tensor_ref = rt_ldr(input_tensor)
#
output_tensor_ref = output_tensor_ref.squeeze(0)
output_tensor_ref = torch.clamp(output_tensor_ref, 0.0, 1.0)
#
to_pil = transforms.ToPILImage()
#
# output_img = to_pil(output_tensor)
# output_img.save("output.png")
#
output_ref_img = to_pil(output_tensor_ref)
output_ref_img.save("output_ref.png")

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
 # print(expected)
