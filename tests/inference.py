from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image(
    path: str | Path,
    channels: int,
) -> torch.Tensor:
    if channels == 3:
        mode = "RGB"
    elif channels == 4:
        mode = "RGBA"
    else:
        raise ValueError("channels must be 3 or 4")

    image = Image.open(path).convert(mode)

    image = Image.open(path).convert(mode)

    data = np.asarray(image).copy()

    tensor = torch.from_numpy(data)
    tensor = tensor.to(dtype=torch.float16)

    # HWC -> CHW
    tensor = tensor.permute(2, 0, 1)

    # CHW -> NCHW
    tensor = tensor.unsqueeze(0)

    # Usually preferable for DNN inputs.
    tensor = tensor / 255.0

    return tensor

def onnx_input_shape(
    program: torch.onnx.ONNXProgram,
) -> list[int | str | None]:
    model = program.model_proto

    for graph_input in model.graph.input:
        if graph_input.name == "input":
            shape = graph_input.type.tensor_type.shape

            result = []

            for dim in shape.dim:
                if dim.HasField("dim_value"):
                    result.append(dim.dim_value)
                elif dim.HasField("dim_param"):
                    result.append(dim.dim_param)
                else:
                    result.append(None)

            return result

    raise ValueError("ONNX input named 'input' not found")


def onnx_example_input_shape(
    program: torch.onnx.ONNXProgram,
    dynamic_default: int = 512,
) -> tuple[int, ...]:
    shape = onnx_input_shape(program)

    return tuple(
        dim if isinstance(dim, int) and dim > 0 else dynamic_default
        for dim in shape
    )

def load_example_image(
    program: torch.onnx.ONNXProgram,
    image_path: str | Path,
) -> torch.Tensor | None:
    shape = onnx_input_shape(program)

    if len(shape) != 4:
        return None

    channels = shape[1]

    if not isinstance(channels, int):
        return None

    if channels not in (3, 4):
        return None

    return load_image(
        image_path,
        channels,
    )


def onnx_infer(
    program: torch.onnx.ONNXProgram,
    image_path: str | Path,
) -> torch.Tensor | None:
    input_tensor = load_example_image(
        program,
        image_path,
    )

    if input_tensor is None:
        return None

    output = program(input_tensor)

    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        if len(output) != 1:
            raise ValueError(
                f"Expected exactly one ONNX output, got {len(output)}"
            )

        if not isinstance(output[0], torch.Tensor):
            raise TypeError(
                f"Expected tensor output, got {type(output[0]).__name__}"
            )

        return output[0]

    raise TypeError(
        f"Expected tensor or single-output tuple/list, got {type(output).__name__}"
    )

def onnx_output_shape(
    program: torch.onnx.ONNXProgram,
) -> list[int | str | None]:
    model = program.model_proto

    for graph_output in model.graph.output:
        if graph_output.name == "output":
            shape = graph_output.type.tensor_type.shape

            result = []

            for dim in shape.dim:
                if dim.HasField("dim_value"):
                    result.append(dim.dim_value)
                elif dim.HasField("dim_param"):
                    result.append(dim.dim_param)
                else:
                    result.append(None)

            return result

    raise ValueError("ONNX output named 'output' not found")

def load_output_image(
    program: torch.onnx.ONNXProgram,
    image_path: str | Path,
) -> torch.Tensor | None:
    shape = onnx_output_shape(program)

    if len(shape) != 4:
        return None

    channels = shape[1]

    if not isinstance(channels, int):
        return None

    if channels not in (3, 4):
        return None

    return load_image(
        image_path,
        channels,
    )
