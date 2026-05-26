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


class TinyPoolUpsampleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_channels = 3
        self.alignment = 16

        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="nearest",
        )

    def forward(self, input):
        return F.relu(self.upsample(F.relu(self.pool(F.relu(input)))))


register_model(TinyPoolUpsampleNet())
