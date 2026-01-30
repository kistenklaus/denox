import io
from typing import BinaryIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.dlpack

INPUT_CHANNELS_COUNT = 3
OUTPUT_CHANNEL_COUNT = 3


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        pm = "zeros"
        ch = INPUT_CHANNELS_COUNT

        gaus_channels = 3

        kernel = (
            torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float16) / 16.0
        )

        self.gaus = nn.Conv2d(
            gaus_channels,
            gaus_channels,
            kernel_size=3,
            padding="same",
            bias=False,
            dtype=torch.float16,
        )

        id_kernel = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float16)

        with torch.no_grad():
            self.gaus.weight.zero_()  # important

            for c in range(gaus_channels):
                self.gaus.weight[c, c] = kernel  # diagonal only

        self.gaus.weight.requires_grad_(False)

        self.enc0 = nn.Conv2d(
            INPUT_CHANNELS_COUNT,
            8,
            3,
            padding="same",
            padding_mode=pm,
            bias=True,
            dtype=torch.float16,
        )
        self.enc1 = nn.Conv2d(
            8, 16, 3, padding="same", padding_mode=pm, bias=True, dtype=torch.float16
        )
        self.enc3 = nn.Conv2d(
            24, 32, 3, padding="same", padding_mode=pm, bias=True, dtype=torch.float16
        )
        # self.con0 = nn.Conv2d(32, 32, 3, padding="same", padding_mode=pm, bias=True, dtype=torch.float16)
        # self.dec0 = nn.Conv2d(32, 16, 3, padding="same", padding_mode=pm)
        #

        self.conv0 = nn.Conv2d(
            INPUT_CHANNELS_COUNT,
            INPUT_CHANNELS_COUNT,
            3,
            padding="same",
            dtype=torch.float16,
            bias=True,
        )

        # conv0_bias = torch.full(
        #     (INPUT_CHANNELS_COUNT,),
        #     0.2,
        #     dtype=torch.float16
        # )
        # with torch.no_grad():
        #     self.conv0.weight.zero_()  # important
        #     self.conv0.bias.copy_(conv0_bias)
            # for c in range(INPUT_CHANNELS_COUNT):
            #     self.conv0.weight[c, c] = id_kernel  # diagonal only

        cha = 64
        chb = 32
        chx = 64

        self.conva = nn.Conv2d(
            INPUT_CHANNELS_COUNT,
            cha,
            3,
            padding="same",
            dtype=torch.float16,
            bias=True,
        )

        self.convb = nn.Conv2d(
            INPUT_CHANNELS_COUNT,
            chb,
            3,
            padding="same",
            dtype=torch.float16,
            bias=True,
        )
        
        self.convx = nn.Conv2d(
            cha + chb,
            chx,
            3,
            padding="same",
            dtype=torch.float16,
            bias=True,
        )

        self.conv0 = nn.Conv2d(
            chx,
            OUTPUT_CHANNEL_COUNT,
            3,
            padding="same",
            dtype=torch.float16,
            bias=False,
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, input):
        H, W = input.size(2), input.size(3)
        alignment = 4  # ensure even H/W so pool+upsample align perfectly
        pad_w = (alignment - (W % alignment)) % alignment
        pad_h = (alignment - (H % alignment)) % alignment
        x = F.pad(input, (0, pad_w, 0, pad_h), mode="replicate")
        # extr = self.enc0(I_aligned)
        # # x_128 = self.pool(extr);
        # x = self.enc0(x)
        #
        # x = pool1 = self.pool(x)
        # x = self.pool(x)
        #
        # x = self.enc1(x)
        #
        # x = self.upsample(input)
        #
        # x = torch.cat((x,pool1), 1)
        #
        # x = self.enc3(x)

        # x0 = self.conv0(input)
        # x1 = self.conv1(x0)
        # x2 = self.conv2(x1)
        # x = torch.cat((x0, x1), 1)
        # x = self.gaus(self.gaus(self.gaus(self.gaus(self.gaus(x)))))
        # x = self.gaus(self.gaus(self.gaus(self.gaus(self.gaus(x)))))
        # x = self.gaus(self.gaus(self.gaus(self.gaus(self.gaus(x)))))
        # x = self.gaus(self.gaus(self.gaus(self.gaus(self.gaus(x)))))
        # x = self.gaus(self.gaus(self.gaus(self.gaus(self.gaus(x)))))
        # x = self.gaus(self.gaus(self.gaus(self.gaus(self.gaus(x)))))
        # x = self.gaus(self.gaus(self.gaus(self.gaus(self.gaus(x)))))
        # x = self.gaus(self.gaus(self.gaus(self.gaus(self.gaus(x)))))

        a = self.conva(x)
        b = self.convb(x)
        ab = torch.cat((a,b), 1)
        x = F.relu(self.convx(ab))
        x = self.conv0(x)

        # x = self.conv2(self.conv8(self.conv1(x)))
        # x = self.conv1(x)
        return x[:, :, :H, :W]


net: nn.Module = Net()
net.to(dtype=torch.float16)

weight0 = torch.ones((32, 3, 3, 3), dtype=torch.float16)
weight1 = torch.ones((32, 32, 3, 3), dtype=torch.float16)

# with torch.no_grad():
#     net.enc0.weight.copy_(weight0)
#     net.con0.weight.copy_(weight1)
#     if net.enc0.bias is not None:
#         net.enc0.bias.fill_(1.0)
#     if net.con0.bias is not None:
#         net.con0.bias.fill_(1.0)

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
