# pydenox/__init__.py
from ._denox import compile as compile, compile_bytes as compile_bytes

def compile_from_torch(
    model,
    example_inputs,
    *,
    onnx_path=None,
    input_names=("input",),
    output_names=("output",),
    dynamic_shapes=None,         # forward to torch.onnx.export if given
    dynamo=True,
    export_params=True,
    external_data=False,
    report=False,
    # any extra torch.onnx.export kwargs can go in export_kwargs
    export_kwargs=None,
    # denox compile kwargs (same as C++ signature)
    **denox_kwargs,
):
    """
    Export a torch.nn.Module to ONNX and compile with denox.
    Returns a denox.Program (same as pydenox.compile).
    """
    import os, tempfile, torch

    export_kwargs = dict(export_kwargs or {})
    export_kwargs.setdefault("dynamo", dynamo)
    export_kwargs.setdefault("export_params", export_params)
    export_kwargs.setdefault("external_data", external_data)
    export_kwargs.setdefault("report", report)
    export_kwargs.setdefault("input_names", list(input_names))
    export_kwargs.setdefault("output_names", list(output_names))
    if dynamic_shapes is not None:
        export_kwargs["dynamic_shapes"] = dynamic_shapes  # PyTorch 2.x

    # Prepare temp ONNX path if user didn't supply one
    cleanup = False
    if onnx_path is None:
        d = tempfile.mkdtemp(prefix="denox_")
        onnx_path = os.path.join(d, "model.onnx")
        cleanup = True

    try:
        model = model.eval()
        # Accept tuple or single tensor for example_inputs
        args = example_inputs if isinstance(example_inputs, (tuple, list)) else (example_inputs,)
        torch.onnx.export(model, args, onnx_path, **export_kwargs)
        return compile(onnx_path, **denox_kwargs)
    finally:
        if cleanup:
            try:
                os.remove(onnx_path)
                os.rmdir(os.path.dirname(onnx_path))
            except OSError:
                pass


