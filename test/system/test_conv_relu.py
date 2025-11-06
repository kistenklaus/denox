import torch
import torch.nn as nn
import torch.nn.functional as F

from test import run_module_test

from denox import Layout

class ConvRelu(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        x = F.relu(self.conv(x))
        return x

def test_conv2d_3x3_3hwc_3hwc_relu():
    run_module_test(
        ConvRelu(nn.Conv2d(3, 3, 3, padding="same", padding_mode="zeros", dtype=torch.float16)),
        torch.rand(1, 3, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_3hwc_32hwc_relu():
    run_module_test(
        ConvRelu(nn.Conv2d(3, 32, 3, padding="same", padding_mode="zeros", dtype=torch.float16)),
        torch.rand(1, 3, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_32hwc_3hwc_relu():
    run_module_test(
        ConvRelu(nn.Conv2d(32, 3, 3, padding="same", padding_mode="zeros", dtype=torch.float16)),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_8hwc_16hwc_relu():
    run_module_test(
        ConvRelu(nn.Conv2d(8, 16, 3, padding="same", padding_mode="zeros", dtype=torch.float16)),
        torch.rand(1, 8, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_16hwc_8hwc_relu():
    run_module_test(
        ConvRelu(nn.Conv2d(16, 8, 3, padding="same", padding_mode="zeros", dtype=torch.float16)),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_16hwc_32hwc_relu():
    run_module_test(
        ConvRelu(nn.Conv2d(16, 32, 3, padding="same", padding_mode="zeros", dtype=torch.float16)),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_32hwc_16hwc_relu_weak():
    run_module_test(
        ConvRelu(nn.Conv2d(32, 16, 3, padding="same", padding_mode="zeros", dtype=torch.float16)),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        atol=1e-2,
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_32hwc_32hwc_relu_weak():
    run_module_test(
        ConvRelu(nn.Conv2d(32, 32, 3, padding="same", padding_mode="zeros", dtype=torch.float16)),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarantee!
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_64hwc_64hwc_relu_weak():
    run_module_test(
        ConvRelu(nn.Conv2d(64, 64, 3, padding="same", padding_mode="zeros", dtype=torch.float16)),
        torch.rand(1, 64, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarnatee!
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


