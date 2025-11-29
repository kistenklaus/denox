import torch
import torch.nn as nn

from test import run_module_test

from denox import Layout


def test_conv2d_3x3_3hwc_3hwc():
    run_module_test(
        nn.Conv2d(3, 3, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 3, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_3hwc_32hwc():
    run_module_test(
        nn.Conv2d(3, 32, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 3, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_32hwc_3hwc():
    run_module_test(
        nn.Conv2d(32, 3, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_8hwc_16hwc():
    run_module_test(
        nn.Conv2d(8, 16, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 8, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_16hwc_8hwc():
    run_module_test(
        nn.Conv2d(16, 8, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_16hwc_32hwc():
    run_module_test(
        nn.Conv2d(16, 32, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_32hwc_16hwc_weak():
    run_module_test(
        nn.Conv2d(32, 16, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        atol=1e-2,
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_conv2d_3x3_24hwc_24hwc():
    run_module_test(
        nn.Conv2d(24, 24, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 24, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_conv2d_3x3_32hwc_32hwc_weak():
    run_module_test(
        nn.Conv2d(32, 32, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarantee!
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_64hwc_64hwc_weak():
    run_module_test(
        nn.Conv2d(64, 64, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 64, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarnatee!
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_128hwc_128hwc_weak():
    run_module_test(
        nn.Conv2d(
            128, 128, 3, padding="same", padding_mode="zeros", dtype=torch.float16
        ),
        torch.rand(1, 128, 100, 100, dtype=torch.float16),
        atol=1e-2,
        rtol=0,
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_256hwc_256hwc_weak():
    run_module_test(
        nn.Conv2d(
            256, 256, 3, padding="same", padding_mode="zeros", dtype=torch.float16
        ),
        torch.rand(1, 256, 100, 100, dtype=torch.float16),
        atol=1e-1,
        rtol=0,
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_8chwc8_16chwc8():
    run_module_test(
        nn.Conv2d(8, 16, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 8, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )


def test_conv2d_3x3_16chwc8_8chwc8():
    run_module_test(
        nn.Conv2d(16, 8, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )


def test_conv2d_3x3_16chwc8_32chwc8():
    run_module_test(
        nn.Conv2d(16, 32, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )


def test_conv2d_3x3_32chwc8_16chwc8_weak():
    run_module_test(
        nn.Conv2d(32, 16, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        atol=1e-2,
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )


def test_conv2d_3x3_32chwc8_32chwc8_weak():
    run_module_test(
        nn.Conv2d(32, 32, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarantee!
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_conv2d_3x3_32chwc8_48chwc8_weak():
    run_module_test(
        nn.Conv2d(32, 48, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarantee!
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_conv2d_3x3_48chwc8_48chwc8_weak():
    run_module_test(
        nn.Conv2d(48, 48, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 48, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarantee!
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )


def test_conv2d_3x3_64chwc8_64chwc8_weak():
    run_module_test(
        nn.Conv2d(64, 64, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 64, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarnatee!
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_conv2d_3x3_64chwc8_80hwc_weak():
    run_module_test(
        nn.Conv2d(64, 80, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 64, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarnatee!
        input_layout=Layout.CHWC8,
        output_layout=Layout.HWC,
    )

def test_conv2d_3x3_80hwc_96hwc_weak():
    run_module_test(
        nn.Conv2d(80, 96, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 80, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarnatee!
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_conv2d_3x3_96hwc_96chwc8_weak():
    run_module_test(
        nn.Conv2d(96, 96, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 96, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarnatee!
        input_layout=Layout.HWC,
        output_layout=Layout.CHWC8,
    )

def test_conv2d_3x3_96chwc8_96chwc8_weak():
    run_module_test(
        nn.Conv2d(96, 96, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 96, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarnatee!
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_conv2d_3x3_128chwc8_64chwc8_weak():
    run_module_test(
        nn.Conv2d(128, 64, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 128, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarnatee!
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_conv2d_3x3_128chwc8_64hwc_weak():
    run_module_test(
        nn.Conv2d(128, 64, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 128, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarnatee!
        input_layout=Layout.CHWC8,
        output_layout=Layout.HWC,
    )

def test_conv2d_3x3_67hwc_64chwc8_weak():
    run_module_test(
        nn.Conv2d(67, 64, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 67, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarnatee!
        input_layout=Layout.HWC,
        output_layout=Layout.CHWC8,
    )

def test_conv2d_3x3_64chwc8_32chwc8_weak():
    run_module_test(
        nn.Conv2d(64, 32, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 64, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarnatee!
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_conv2d_3x3_32chwc8_3hwc_weak():
    run_module_test(
        nn.Conv2d(32, 3, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarnatee!
        input_layout=Layout.CHWC8,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_128chwc8_128chwc8_weak():
    run_module_test(
        nn.Conv2d(
            128, 128, 3, padding="same", padding_mode="zeros", dtype=torch.float16
        ),
        torch.rand(1, 128, 100, 100, dtype=torch.float16),
        atol=1e-2,
        rtol=0,
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_conv2d_3x3_160chwc8_112chwc8_weak():
    run_module_test(
        nn.Conv2d(160, 112, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 160, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarnatee!
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_conv2d_3x3_112chwc8_112chwc8_weak():
    run_module_test(
        nn.Conv2d(112, 112, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 112, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarnatee!
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )

def test_conv2d_3x3_160chwc8_96hwc_weak():
    run_module_test(
        nn.Conv2d(160, 96, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 160, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarnatee!
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )


def test_conv2d_3x3_256chwc8_256chwc8_weak():
    run_module_test(
        nn.Conv2d(
            256, 256, 3, padding="same", padding_mode="zeros", dtype=torch.float16
        ),
        torch.rand(1, 256, 100, 100, dtype=torch.float16),
        atol=1e-1,
        rtol=0,
        input_layout=Layout.CHWC8,
        output_layout=Layout.CHWC8,
    )


def test_conv2d_3x3_8hwc_16chwc8():
    run_module_test(
        nn.Conv2d(8, 16, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 8, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.CHWC8,
    )


def test_conv2d_3x3_16hwc_8chwc8():
    run_module_test(
        nn.Conv2d(16, 8, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.CHWC8,
    )


def test_conv2d_3x3_16hwc_32chwc8_weak():
    run_module_test(
        nn.Conv2d(16, 32, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.CHWC8,
        atol=1e-2,
    )


def test_conv2d_3x3_32hwc_16chwc8_weak():
    run_module_test(
        nn.Conv2d(32, 16, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        atol=1e-2,
        input_layout=Layout.HWC,
        output_layout=Layout.CHWC8,
    )


def test_conv2d_3x3_32hwc_32chwc8_weak():
    run_module_test(
        nn.Conv2d(32, 32, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarantee!
        input_layout=Layout.HWC,
        output_layout=Layout.CHWC8,
    )



def test_conv2d_3x3_64hwc_64chwc8_weak():
    run_module_test(
        nn.Conv2d(64, 64, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 64, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarnatee!
        input_layout=Layout.HWC,
        output_layout=Layout.CHWC8,
    )


def test_conv2d_3x3_128hwc_128chwc8_weak():
    run_module_test(
        nn.Conv2d(
            128, 128, 3, padding="same", padding_mode="zeros", dtype=torch.float16
        ),
        torch.rand(1, 128, 100, 100, dtype=torch.float16),
        atol=1e-2,
        rtol=0,
        input_layout=Layout.HWC,
        output_layout=Layout.CHWC8,
    )


def test_conv2d_3x3_256hwc_256chwc8_weak():
    run_module_test(
        nn.Conv2d(
            256, 256, 3, padding="same", padding_mode="zeros", dtype=torch.float16
        ),
        torch.rand(1, 256, 100, 100, dtype=torch.float16),
        atol=1e-1,
        rtol=0,
        input_layout=Layout.HWC,
        output_layout=Layout.CHWC8,
    )


def test_conv2d_3x3_8chwc8_16hwc():
    run_module_test(
        nn.Conv2d(8, 16, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 8, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_16chwc8_8hwc():
    run_module_test(
        nn.Conv2d(16, 8, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_16chwc8_32hwc_weak():
    run_module_test(
        nn.Conv2d(16, 32, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 16, 100, 100, dtype=torch.float16),
        input_layout=Layout.CHWC8,
        output_layout=Layout.HWC,
        atol=1e-2,
    )


def test_conv2d_3x3_32chwc8_16hwc_weak():
    run_module_test(
        nn.Conv2d(32, 16, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        atol=1e-2,
        input_layout=Layout.CHWC8,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_32chwc8_32hwc_weak():
    run_module_test(
        nn.Conv2d(32, 32, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 32, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarantee!
        input_layout=Layout.CHWC8,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_64chwc8_64hwc_weak():
    run_module_test(
        nn.Conv2d(64, 64, 3, padding="same", padding_mode="zeros", dtype=torch.float16),
        torch.rand(1, 64, 100, 100, dtype=torch.float16),
        atol=1e-2,  # weaker guarnatee!
        input_layout=Layout.CHWC8,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_128chwc8_128hwc_weak():
    run_module_test(
        nn.Conv2d(
            128, 128, 3, padding="same", padding_mode="zeros", dtype=torch.float16
        ),
        torch.rand(1, 128, 100, 100, dtype=torch.float16),
        atol=1e-2,
        rtol=0,
        input_layout=Layout.CHWC8,
        output_layout=Layout.HWC,
    )


def test_conv2d_3x3_256chwc8_256hwc_weak():
    run_module_test(
        nn.Conv2d(
            256, 256, 3, padding="same", padding_mode="zeros", dtype=torch.float16
        ),
        torch.rand(1, 256, 100, 100, dtype=torch.float16),
        atol=1e-1,
        rtol=0,
        input_layout=Layout.CHWC8,
        output_layout=Layout.HWC,
    )

