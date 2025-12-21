import io
from typing import BinaryIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.dlpack
import denox
# from denox import DataType, Layout, Module, Shape, Storage, TargetEnv

INPUT_CHANNELS_COUNT = 3
OUTPUT_CHANNEL_COUNT = 16


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(
            INPUT_CHANNELS_COUNT,  # albedo + norm
            INPUT_CHANNELS_COUNT,
            kernel_size=3,
            padding="same",
            dtype=torch.float16,
        )

        self.conv1 = nn.Conv2d(
            INPUT_CHANNELS_COUNT * 2,  # albedo + norm
            OUTPUT_CHANNEL_COUNT,
            kernel_size=3,
            padding="same",
            dtype=torch.float16,
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, albedo, norm):
        albedo = self.conv0(albedo)
        norm = self.conv0(norm)

        albedo = self.pool(albedo)
        norm = self.pool(norm)

        albedo = self.upsample(albedo)
        norm = self.upsample(norm)

        x = torch.cat((albedo, norm), dim=1)

        H, W = x.size(2), x.size(3)
        alignment = 4
        pad_w = (alignment - (W % alignment)) % alignment
        pad_h = (alignment - (H % alignment)) % alignment
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        x = self.conv1(x)
        return x[:, :, :H, :W]


example_albedo = torch.ones(1, INPUT_CHANNELS_COUNT, 5, 5, dtype=torch.float16)
example_norm = torch.ones(1, INPUT_CHANNELS_COUNT, 5, 5, dtype=torch.float16)

net: nn.Module = Net()

program = torch.onnx.export(
    net,
    (example_albedo, example_norm),  # tuple of inputs
    input_names=["albedo", "norm"],
    output_names=["output"],
    dynamic_axes={
        "albedo": {2: "H", 3: "W"},
        "norm": {2: "H", 3: "W"},
        "output": {2: "H", 3: "W"},
    },
)

program.save("net.onnx")
