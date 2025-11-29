import torch
import torch.nn as nn

from test import run_module_test

from denox import Layout


def test_upsample_nearest_3hwc():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 3, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_upsample_nearest_8hwc():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 8, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_upsample_nearest_9hwc():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 9, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_upsample_nearest_16hwc():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_upsample_nearest_24hwc():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 24, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_upsample_nearest_32hwc():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_upsample_nearest_48hwc():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 48, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_upsample_nearest_64hwc():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 64, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_upsample_nearest_96hwc():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 96, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_upsample_nearest_112hwc():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 112, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_upsample_nearest_8chwc8():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 8, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_upsample_nearest_16chwc8():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_upsample_nearest_24chwc8():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 24, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_upsample_nearest_32chwc8():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_upsample_nearest_48chwc8():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 48, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_upsample_nearest_64chwc8():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 64, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_upsample_nearest_96chwc8():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 96, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_upsample_nearest_112chwc8():
    run_module_test(
        nn.Upsample(scale_factor=2),
        torch.rand(1, 112, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )
