from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.jit import Error
import torch.nn as nn
import torch.nn.functional as F


MODELS: list[torch.onnx.ONNXProgram] = []


def register_model(
    model: nn.Module
) -> None:
    input_channels = int(model.input_channels)

    input_height = getattr(model, "input_height", None)
    input_width = getattr(model, "input_width", None)

    if input_height is not None:
        input_height = int(input_height)

        if input_height <= 0:
            raise ValueError("model.input_height must be positive")

    if input_width is not None:
        input_width = int(input_width)

        if input_width <= 0:
            raise ValueError("model.input_width must be positive")

    example_height = input_height or 64
    example_width = input_width or 64

    dynamic_input_shape: dict[int, torch.export.Dim] = {}

    if input_height is None:
        dynamic_input_shape[2] = torch.export.Dim.DYNAMIC

    if input_width is None:
        dynamic_input_shape[3] = torch.export.Dim.DYNAMIC

    dynamic_shapes = None

    if dynamic_input_shape:
        dynamic_shapes = {
            "input": dynamic_input_shape,
        }

    alignment = getattr(model, "alignment", None)

    if alignment is not None:
        alignment = int(alignment)

        if alignment <= 0:
            raise ValueError("model.alignment must be positive")

        class AlignedModel(nn.Module):
            def __init__(self, wrapped_model: nn.Module):
                super().__init__()

                self.wrapped_model = wrapped_model

            def forward(self, input):
                h = input.size(2)
                w = input.size(3)

                pad_w = (alignment - (w % alignment)) % alignment
                pad_h = (alignment - (h % alignment)) % alignment

                aligned = F.pad(
                    input,
                    (0, pad_w, 0, pad_h),
                    mode="replicate",
                )

                output = self.wrapped_model(aligned)

                return output[:, :, :h, :w]

        model = AlignedModel(model)

    model.eval()
    model.to(dtype=torch.float16)

    example_input = torch.ones(
        (
            1,
            input_channels,
            example_height,
            example_width,
        ),
        dtype=torch.float16,
    )

    with torch.no_grad():
        program = torch.onnx.export(
            model,
            (example_input,),
            f=None,
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
        )
    if program is None:
        raise Error("Failed to export")

    MODELS.append(program)


class OIDNNet(nn.Module):
    def __init__(self, in_channels, out_channels, small=False):
        super(OIDNNet, self).__init__()
        self.input_channels = in_channels

        # Number of channels per layer
        ic = in_channels
        if small:
            ec1 = 32
            ec2 = 32
            ec3 = 32
            ec4 = 32
            ec5 = 32
            dc4 = 64
            dc3 = 64
            dc2a = 64
            dc2b = 32
            dc1a = 32
            dc1b = 32
        else:
            ec1 = 32
            ec2 = 48
            ec3 = 64
            ec4 = 80
            ec5 = 96
            dc4 = 112
            dc3 = 96
            dc2a = 64
            dc2b = 64
            dc1a = 64
            dc1b = 32
        oc = out_channels


        def Conv(in_channels, out_channels, bias=True):
            return nn.Conv2d(in_channels, out_channels, 3, padding="same", padding_mode="zeros", dtype=torch.float16,bias=bias)
        # Convolutions
        self.enc_conv0 = Conv(ic, ec1)
        self.enc_conv1 = Conv(ec1, ec1)
        self.enc_conv2 = Conv(ec1, ec2)
        self.enc_conv3 = Conv(ec2, ec3)
        self.enc_conv4 = Conv(ec3, ec4)
        self.enc_conv5a = Conv(ec4, ec5)
        self.enc_conv5b = Conv(ec5, ec5)
        self.dec_conv4a = Conv(ec5 + ec3, dc4)
        self.dec_conv4b = Conv(dc4, dc4)
        self.dec_conv3a = Conv(dc4 + ec2, dc3)
        self.dec_conv3b = Conv(dc3, dc3)
        self.dec_conv2a = Conv(dc3 + ec1, dc2a)
        self.dec_conv2b = Conv(dc2a, dc2b)
        self.dec_conv1a = Conv(dc2b + ic, dc1a)
        self.dec_conv1b = Conv(dc1a, dc1b)
        self.dec_conv0 = Conv(dc1b, oc, bias=False)

        # Images must be padded to multiples of the alignment
        self.alignment = 16

    def forward(self, input):
        def pool(x):
            return F.max_pool2d(x, 2, 2)
        def upsample(x):
            return F.interpolate(x, scale_factor=2, mode="nearest")
        def concat(a, b):
            return torch.cat((a, b), 1)
        # Encoder
        # -------------------------------------------

        x = F.relu(self.enc_conv0(input))  # enc_conv0

        x = F.relu(self.enc_conv1(x))  # enc_conv1
        x = pool1 = pool(x)  # pool1

        x = F.relu(self.enc_conv2(x))  # enc_conv2
        x = pool2 = pool(x)  # pool2

        x = F.relu(self.enc_conv3(x))  # enc_conv3
        x = pool3 = pool(x)  # pool3

        x = F.relu(self.enc_conv4(x))  # enc_conv4
        x = pool(x)  # pool4

        # Bottleneck
        x = F.relu(self.enc_conv5a(x))  # enc_conv5a
        x = F.relu(self.enc_conv5b(x))  # enc_conv5b

        # Decoder
        # -------------------------------------------

        x = upsample(x)  # upsample4
        x = concat(x, pool3)  # concat4
        x = F.relu(self.dec_conv4a(x))  # dec_conv4a
        x = F.relu(self.dec_conv4b(x))  # dec_conv4b

        x = upsample(x)  # upsample3
        x = concat(x, pool2)  # concat3
        x = F.relu(self.dec_conv3a(x))  # dec_conv3a
        x = F.relu(self.dec_conv3b(x))  # dec_conv3b

        x = upsample(x)  # upsample2
        x = concat(x, pool1)  # concat2
        x = F.relu(self.dec_conv2a(x))  # dec_conv2a
        x = F.relu(self.dec_conv2b(x))  # dec_conv2b

        x = upsample(x)  # upsample1
        x = concat(x, input)  # concat1
        x = F.relu(self.dec_conv1a(x))  # dec_conv1a
        x = F.relu(self.dec_conv1b(x))  # dec_conv1b

        x = self.dec_conv0(x)  # dec_conv0

        return x

register_model(OIDNNet(3,3, True))
register_model(OIDNNet(3,3, False))

class ManyConv(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_channels = 3
        self.output_channels = 3
        self.ch = 32

        self.alignment = 1

        self.conv0 = nn.Conv2d(
            self.input_channels,
            self.ch,
            3,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dtype=torch.float16,
        )

        self.conv1a = nn.Conv2d(
            self.ch,
            self.ch,
            3,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dtype=torch.float16,
        )
        self.conv1b = nn.Conv2d(
            self.ch,
            self.ch,
            3,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dtype=torch.float16,
        )

        self.conv2a = nn.Conv2d(
            self.ch,
            self.ch,
            3,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dtype=torch.float16,
        )

        self.conv2b = nn.Conv2d(
            self.ch,
            self.ch,
            3,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dtype=torch.float16,
        )

        self.conv3 = nn.Conv2d(
            self.ch,
            self.output_channels,
            3,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dtype=torch.float16,
        )


    def forward(self, input):
        x = F.relu(self.conv0(input))
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = F.relu(self.conv3(x))
        return x

register_model(ManyConv())

class YetAnotherUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_channels = 3
        # pm = 'reflect' # not implemented you suckers
        pm = "zeros"
        self.enc0 = nn.Conv2d(
            self.input_channels, 32, 3, padding="same", padding_mode=pm
        )
        self.enc1 = nn.Conv2d(32, 48, 3, padding="same", padding_mode=pm)
        self.enc2 = nn.Conv2d(48, 64, 3, padding="same", padding_mode=pm)
        self.enc3 = nn.Conv2d(64, 80, 3, padding="same", padding_mode=pm)
        self.enc4 = nn.Conv2d(80, 112, 3, padding="same", padding_mode=pm)
        self.enc5 = nn.Conv2d(112, 112, 3, padding="same", padding_mode=pm)

        # self.extr = nn.Conv2d(32, 32, 3, padding='same', padding_mode=pm)

        self.dec0 = nn.Conv2d(112 + 112, 112, 3, padding="same", padding_mode=pm)
        self.con0 = nn.Conv2d(112, 112, 3, padding="same", padding_mode=pm)
        self.dec1 = nn.Conv2d(112 + 80, 80, 3, padding="same", padding_mode=pm)
        self.con1 = nn.Conv2d(80, 80, 3, padding="same", padding_mode=pm)
        self.dec2 = nn.Conv2d(80 + 64, 64, 3, padding="same", padding_mode=pm)
        self.con2 = nn.Conv2d(64, 64, 3, padding="same", padding_mode=pm)
        self.dec3 = nn.Conv2d(64 + 48, 48, 3, padding="same", padding_mode=pm)
        self.con3 = nn.Conv2d(48, 48, 3, padding="same", padding_mode=pm)
        self.dec4 = nn.Conv2d(48 + 32, 16, 3, padding="same", padding_mode=pm)
        self.con4 = nn.Conv2d(16, 16, 3, padding="same", padding_mode=pm)
        self.dec5 = nn.Conv2d(
            16 + self.input_channels, 12, 3, padding="same", padding_mode=pm
        )
        self.con5 = nn.Conv2d(12, 12, 3, padding="same", padding_mode=pm)
        # self.dec5 = nn.Conv2d(12+32, 12, 3, padding='same', padding_mode=pm)

        # as much as i hate it, these extra convolutions seem to contribute quite a bit to image sharpness/overall fidelity.
        # potentially a more clever upsampling/convolution directly there would make the architecture more lightweight.

        # self.con0a = nn.Conv2d(101, 101, 3, padding='same', padding_mode=pm)
        # self.con1a = nn.Conv2d(76, 76, 3, padding='same', padding_mode=pm)
        # self.con2a = nn.Conv2d(57, 57, 3, padding='same', padding_mode=pm)
        # self.con3a = nn.Conv2d(43, 43, 3, padding='same', padding_mode=pm)
        # self.con4a = nn.Conv2d(16, 16, 3, padding='same', padding_mode=pm)
        # self.con5a = nn.Conv2d(12, 12, 3, padding='same', padding_mode=pm)

        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_out = nn.Conv2d(12,3,3,padding="same", padding_mode=pm)

        self.alignment = 64

    def forward(self, input):
        extr = F.relu(self.enc0(input))
        x_128 = self.pool(extr)  # self.pool(F.relu(self.extr(extr)))
        # x_128 = self.pool(F.relu(self.enc0(I)))
        x_64 = self.pool(F.relu(self.enc1(x_128)))
        x_32 = self.pool(F.relu(self.enc2(x_64)))
        x_16 = self.pool(F.relu(self.enc3(x_32)))
        x_8 = self.pool(F.relu(self.enc4(x_16)))
        x_4 = self.pool(F.relu(self.enc5(x_8)))

        x = F.relu(self.dec0(torch.cat([self.upsample(x_4), x_8], 1)))
        x = F.relu(self.con0(x))
        # x     = F.relu(self.con0a(x))
        # x     = F.relu(self.dec1(torch.cat([self.upsample(x_8), x_16],  1)))
        x = F.relu(self.dec1(torch.cat([self.upsample(x), x_16], 1)))
        x = F.relu(self.con1(x))
        # x     = F.relu(self.con1a(x))
        x = F.relu(self.dec2(torch.cat([self.upsample(x), x_32], 1)))
        x = F.relu(self.con2(x))
        # x     = F.relu(self.con2a(x))
        x = F.relu(self.dec3(torch.cat([self.upsample(x), x_64], 1)))
        x = F.relu(self.con3(x))
        # x     = F.relu(self.con3a(x))
        x = F.relu(self.dec4(torch.cat([self.upsample(x), x_128], 1)))
        # x     = F.relu(self.dec5(torch.cat([self.upsample(x),   extr],  1)))
        x = F.relu(self.con4(x))
        # x     = F.relu(self.con4a(x))
        x = F.relu(self.dec5(torch.cat([self.upsample(x), input], 1)))
        x = F.relu(self.con5(x))
        # x     = F.relu(self.con5a(x))

        return self.conv_out(x)

register_model(YetAnotherUNet())


