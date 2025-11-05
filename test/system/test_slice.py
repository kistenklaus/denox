import torch
import torch.nn as nn

from test import run_module_test

from denox import Layout


class SliceNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, :, 20:40, 10:30]


def test_slice_3hwc():
    run_module_test(
        SliceNet(),
        torch.rand(1, 3, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_slice_8hwc():
    run_module_test(
        SliceNet(),
        torch.rand(1, 8, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_slice_9hwc():
    run_module_test(
        SliceNet(),
        torch.rand(1, 9, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_slice_16hwc():
    run_module_test(
        SliceNet(),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_slice_24hwc():
    run_module_test(
        SliceNet(),
        torch.rand(1, 24, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_slice_32hwc():
    run_module_test(
        SliceNet(),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_slice_48hwc():
    run_module_test(
        SliceNet(),
        torch.rand(1, 48, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_slice_64hwc():
    run_module_test(
        SliceNet(),
        torch.rand(1, 48, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )
