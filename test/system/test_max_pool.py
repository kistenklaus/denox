import torch
import torch.nn as nn

from test import run_module_test

from denox import Layout

def test_max_pool_2x2_3hwc():
    run_module_test(
        nn.MaxPool2d(2,2),
        torch.rand(1, 3, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_max_pool_2x2_8hwc():
    run_module_test(
        nn.MaxPool2d(2,2),
        torch.rand(1, 8, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_max_pool_2x2_9hwc():
    run_module_test(
        nn.MaxPool2d(2,2),
        torch.rand(1, 9, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_max_pool_2x2_16hwc():
    run_module_test(
        nn.MaxPool2d(2,2),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_max_pool_2x2_24hwc():
    run_module_test(
        nn.MaxPool2d(2,2),
        torch.rand(1, 24, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_max_pool_2x2_32hwc():
    run_module_test(
        nn.MaxPool2d(2,2),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_max_pool_2x2_48hwc():
    run_module_test(
        nn.MaxPool2d(2,2),
        torch.rand(1, 48, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_max_pool_2x2_64hwc():
    run_module_test(
        nn.MaxPool2d(2,2),
        torch.rand(1, 64, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_max_pool_2x2_80hwc():
    run_module_test(
        nn.MaxPool2d(2,2),
        torch.rand(1, 80, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_max_pool_2x2_8chwc8():
    run_module_test(
        nn.MaxPool2d(2,2),
        torch.rand(1, 8, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_max_pool_2x2_16chwc8():
    run_module_test(
        nn.MaxPool2d(2,2),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_max_pool_2x2_24chwc8():
    run_module_test(
        nn.MaxPool2d(2,2),
        torch.rand(1, 24, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_max_pool_2x2_32chwc8():
    run_module_test(
        nn.MaxPool2d(2,2),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_max_pool_2x2_48chwc8():
    run_module_test(
        nn.MaxPool2d(2,2),
        torch.rand(1, 48, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_max_pool_2x2_64chwc8():
    run_module_test(
        nn.MaxPool2d(2,2),
        torch.rand(1, 64, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_max_pool_2x2_80chwc8():
    run_module_test(
        nn.MaxPool2d(2,2),
        torch.rand(1, 80, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )
