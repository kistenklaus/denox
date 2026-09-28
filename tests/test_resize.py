from helpers import run_denox
from models import RESIZE_MODELS
from images import create_random_png
from inference import onnx_infer, load_output_image, onnx_input_shape
from onnx import TensorProto, helper, numpy_helper
import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


SIZES = [(640, 360), (257, 131)]


@pytest.mark.parametrize("index", range(len(RESIZE_MODELS)))
def test_resize(cache_dir, tmp_path, index):
    program = RESIZE_MODELS[index]
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    db_path = cache_dir / "gpu.db"

    program.save(onnx_path)
    output = run_denox(
        "compile", onnx_path,
        f"--db={db_path}",
        "-o", dnx_path,
        "--min-samples=2",
        "--max-samples=2",
        "--relative-error=1",
        verbose=True,
    )
    assert output.returncode == 0

    sizes = SIZES
    _, _, h, w = onnx_input_shape(program)
    if isinstance(h, int) and isinstance(w, int):
        sizes = [(w, h)]

    for width, height in sizes:
        create_random_png(input_path, width, height)
        output = run_denox(
            "infer", dnx_path,
            "-i", input_path,
            "-o", output_path,
            verbose=True,
        )
        assert output.returncode == 0
        ref = onnx_infer(program, input_path)
        assert ref is not None
        image = load_output_image(program, output_path)
        assert image is not None
        assert ref.shape == image.shape
        torch.testing.assert_close(
            image.to(torch.float32),
            ref.to(torch.float32).clamp(0, 1),
            rtol=0,
            atol=1e-2,
        )


class BicubicUpsample(nn.Module):
    def forward(self, input):
        return F.interpolate(input, scale_factor=2, mode="bicubic")


def test_resize_unsupported_mode(tmp_path):
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"

    with torch.no_grad():
        program = torch.onnx.export(
            BicubicUpsample(),
            (torch.ones(1, 3, 64, 64),),
            f=None,
            input_names=["input"],
            output_names=["output"],
            dynamo=True,
        )
    assert program is not None
    program.save(onnx_path)
    output = run_denox(
        "compile", onnx_path,
        "-o", dnx_path,
        verbose=False,
    )
    assert output.returncode != 0
    assert "mode=\"linear\"" in output.stdout + output.stderr


def make_resize_model(scales=None, sizes=None, dynamic=False, **attributes):
    height, width = ("H", "W") if dynamic else (16, 16)
    initializers = []
    inputs = ["input", ""]
    if scales is not None:
        initializers.append(numpy_helper.from_array(
            np.array(scales, np.float32), "scales"))
    inputs.append("scales" if scales is not None else "")
    if sizes is not None:
        initializers.append(numpy_helper.from_array(
            np.array(sizes, np.int64), "sizes"))
        inputs.append("sizes")
    node = helper.make_node("Resize", inputs, ["output"], **attributes)
    graph = helper.make_graph(
        [node], "resize",
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT16, [1, 3, height, width])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT16, [1, 3, None, None])],
        initializers,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 19)])


RESIZE_CASES = {
    "linear_scales": (dict(scales=[1, 1, 2, 2]), None),
    "linear_sizes": (dict(sizes=[1, 3, 32, 32]), None),
    "asymmetric": (
        dict(scales=[1, 1, 2, 2],
             coordinate_transformation_mode="asymmetric"),
        "only supports coordinate_transformation_mode",
    ),
    "sizes_not_multiple": (
        dict(sizes=[1, 3, 24, 24]),
        "sizes must be a static integer multiple",
    ),
    "sizes_dynamic_input": (
        dict(sizes=[1, 3, 32, 32], dynamic=True),
        "sizes must be a static integer multiple",
    ),
    "sizes_anisotropic": (
        dict(sizes=[1, 3, 32, 48]),
        "only isotropic upsampling",
    ),
    "scales_anisotropic": (
        dict(scales=[1, 1, 2, 3]),
        "only isotropic upsampling",
    ),
    "antialias": (
        dict(scales=[1, 1, 2, 2], antialias=1),
        "antialias not supported",
    ),
    "keep_aspect_ratio_policy": (
        dict(sizes=[1, 3, 32, 32], keep_aspect_ratio_policy="not_larger"),
        "keep_aspect_ratio_policy",
    ),
}


@pytest.mark.parametrize("case", RESIZE_CASES)
def test_resize_linear_attributes(tmp_path, case):
    kwargs, message = RESIZE_CASES[case]
    onnx_path = tmp_path / "net.onnx"
    dnx_path = tmp_path / "net.dnx"

    onnx.save(make_resize_model(mode="linear", **kwargs), onnx_path)
    output = run_denox(
        "compile", onnx_path,
        "-o", dnx_path,
        verbose=False,
    )
    if message is None:
        assert output.returncode == 0
    else:
        assert output.returncode != 0
        assert message in output.stdout + output.stderr
