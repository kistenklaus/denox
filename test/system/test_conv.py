import torch
import torch.nn as nn

from test import run_module_test


def test_conv2d_3_3_3():
    run_module_test(
        nn.Conv2d(3, 3, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 3, 64, 64, dtype=torch.float16),
    )

def test_conv2d_3_32_3():
    run_module_test(
        nn.Conv2d(3, 32, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 3, 64, 64, dtype=torch.float16),
    )

def test_conv2d_8_16_3():
    run_module_test(
        nn.Conv2d(8, 16, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 8, 64, 64, dtype=torch.float16),
        atol=1e-3
    )

def test_conv2d_16_32_3():
    in_channels = 16
    out_channels = 32
    weight = torch.ones((out_channels, in_channels, 3, 3), dtype=torch.float16)
    conv = nn.Conv2d(in_channels, out_channels, 3, padding="same", padding_mode="zeros", dtype=torch.float16, bias=False)
    with torch.no_grad():
        conv.weight.copy_(weight)

    run_module_test(
        conv,
        torch.ones(1, in_channels, 4, 4, dtype=torch.float16),
        atol=1e-3,
    )

def test_conv2d_32_32_3():
    run_module_test(
        nn.Conv2d(32, 32, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 32, 64, 64, dtype=torch.float16),
        atol=1e-2,
        rtol=0,
    )



def test_conv2d_64_64_3():
    run_module_test(
        nn.Conv2d(32, 32, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 32, 64, 64, dtype=torch.float16),
        atol=1e-2,
        rtol=0,
    )


def test_conv2d_128_128_3():
    run_module_test(
        nn.Conv2d(32, 32, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 32, 64, 64, dtype=torch.float16),
        atol=1e-2,
        rtol=0,
    )
