import torch
import torch.nn as nn
import torch.nn.functional as F

from test import run_module_test

from denox import Layout


class PadNet(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        x = F.pad(x, (100, 200, 300, 400), mode=self.mode)
        return x


def test_pad_3hwc():
    run_module_test(
        PadNet("replicate"),
        torch.rand(1, 3, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_pad_8hwc():
    run_module_test(
        PadNet("replicate"),
        torch.rand(1, 8, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_pad_9hwc():
    run_module_test(
        PadNet("replicate"),
        torch.rand(1, 9, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_pad_16hwc():
    run_module_test(
        PadNet("replicate"),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_pad_24hwc():
    run_module_test(
        PadNet("replicate"),
        torch.rand(1, 24, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_pad_32hwc():
    run_module_test(
        PadNet("replicate"),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_pad_48hwc():
    run_module_test(
        PadNet("replicate"),
        torch.rand(1, 48, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_pad_64hwc():
    run_module_test(
        PadNet("replicate"),
        torch.rand(1, 64, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


