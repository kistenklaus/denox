import torch
import torch.nn as nn
import torch.nn.functional as F

from test import run_module_test

from denox import Layout

class ConcatNet(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, 3, padding="same", padding_mode="zeros", dtype=torch.float16)

    def forward(self, x):
        y = self.conv(x)
        x = torch.cat([y,x], 1)
        return x


def test_concat_3hwc():
    run_module_test(
        ConcatNet(3),
        torch.rand(1, 3, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_concat_8hwc():
    run_module_test(
        ConcatNet(8),
        torch.rand(1, 8, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_concat_9hwc():
    run_module_test(
        ConcatNet(9),
        torch.rand(1, 9, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_concat_16hwc():
    run_module_test(
        ConcatNet(16),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_concat_24hwc():
    run_module_test(
        ConcatNet(24),
        torch.rand(1, 24, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_concat_32hwc():
    run_module_test(
        ConcatNet(32),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
        atol=1e-2,
    )

def test_concat_48hwc():
    run_module_test(
        ConcatNet(48),
        torch.rand(1, 48, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
        atol=1e-2,
    )


def test_concat_8chwc8():
    run_module_test(
        ConcatNet(8),
        torch.rand(1, 8, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_concat_16chwc8():
    run_module_test(
        ConcatNet(16),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_concat_24chwc8():
    run_module_test(
        ConcatNet(24),
        torch.rand(1, 24, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_concat_32chwc8():
    run_module_test(
        ConcatNet(32),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
        atol=1e-2,
    )

def test_concat_48chwc8():
    run_module_test(
        ConcatNet(48),
        torch.rand(1, 48, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
        atol=1e-2,
    )
