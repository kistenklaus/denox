import torch
import torch.nn as nn
import torch.nn.functional as F

from test import run_module_test

from denox import Layout

class ReluNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x)

def test_relu_3hwc():
    run_module_test(
        ReluNet(),
        torch.rand(1, 3, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_relu_8hwc():
    run_module_test(
        ReluNet(),
        torch.rand(1, 8, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_relu_9hwc():
    run_module_test(
        ReluNet(),
        torch.rand(1, 9, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_relu_16hwc():
    run_module_test(
        ReluNet(),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_relu_24hwc():
    run_module_test(
        ReluNet(),
        torch.rand(1, 24, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_relu_32hwc():
    run_module_test(
        ReluNet(),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_relu_48hwc():
    run_module_test(
        ReluNet(),
        torch.rand(1, 48, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_relu_64hwc():
    run_module_test(
        ReluNet(),
        torch.rand(1, 64, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_relu_8chwc8():
    run_module_test(
        ReluNet(),
        torch.rand(1, 8, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_relu_16chwc8():
    run_module_test(
        ReluNet(),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_relu_24chwc8():
    run_module_test(
        ReluNet(),
        torch.rand(1, 24, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_relu_32chwc8():
    run_module_test(
        ReluNet(),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_relu_48chwc8():
    run_module_test(
        ReluNet(),
        torch.rand(1, 48, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_relu_64chwc8():
    run_module_test(
        ReluNet(),
        torch.rand(1, 64, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

