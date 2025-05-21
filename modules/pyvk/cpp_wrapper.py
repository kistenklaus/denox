from typing import List
import _pyvk_cpp as cpp
from .reflect import ReflectedLayer
import torch
from enum import Enum
import logging

# Set up logging for warnings
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CppWrapper")


def generate_pipeline(reflected_layers: List[ReflectedLayer]) -> None:
    native_layers = []

    for layer in reflected_layers:
        native = {
            "name": layer.name,
            "type": layer.layer_type.value,
            "input_shape": layer.input_shape,
            "output_shape": layer.output_shape,
            "parameters": {},
        }

        for key, value in layer.parameters.items():
            key_str = key.value  # use enum .value instead of .name
            if isinstance(value, torch.Tensor):
                native["parameters"][key_str] = value.tolist()
            elif isinstance(value, Enum):
                native["parameters"][key_str] = value.value
            else:
                native["parameters"][key_str] = value

        native_layers.append(native)

    pipelineDescription = cpp.generatePipeline(native_layers)
    if pipelineDescription["error"] != None:
        logger.error(f"pyvk_cpp Error: {pipelineDescription["error"]}");
        return None

    pipelineDescription = pipelineDescription["value"] # unwrap
    
    print(pipelineDescription)

    return None
