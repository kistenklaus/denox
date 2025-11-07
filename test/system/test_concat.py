import torch
import torch.nn as nn
import torch.nn.functional as F

from test import run_module_test

from denox import Layout

class ConcatNet(nn.Module):
    def __init__(self, ich, ch1, ch2):
        super().__init__()
        self.conv0 = nn.Conv2d(ich, ch1 , 3, padding="same", padding_mode="zeros", dtype=torch.float16)
        self.conv1 = nn.Conv2d(ich, ch2 , 3, padding="same", padding_mode="zeros", dtype=torch.float16)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x = torch.cat((x0,x1), 1)
        return x


def test_concat_3hwc():
    run_module_test(
        ConcatNet(3, 3, 3),
        torch.rand(1, 3, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )

def test_concat_3hwc_4hwc():
    run_module_test(
        ConcatNet(3, 3, 4),
        torch.rand(1, 3, 2, 2, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
    )
