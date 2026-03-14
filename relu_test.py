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
        self.alignment=16
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, input):
        return F.relu(self.upsample(F.relu(self.pool(F.relu(input)))))

class UNetAlignment(nn.Module):
    def __init__(self, net):
        super(UNetAlignment, self).__init__()
        self.net = net

    def forward(self, input):
        alignment = self.net.alignment  # ensure even H/W so pool+upsample align perfectly
        H, W = input.size(2), input.size(3)
        pad_w = (alignment - (W % alignment)) % alignment
        pad_h = (alignment - (H % alignment)) % alignment
        aligned = F.pad(input, (0, pad_w, 0, pad_h), mode="replicate")
        output = self.net(aligned)
        return output[:,:,:H,:W]

net: nn.Module = UNetAlignment(Net())
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
