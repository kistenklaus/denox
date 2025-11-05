import torch
import torch.nn as nn

from test import run_module_test

from denox import Layout


class SliceNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, :, :40, :40]


# def test_slice_3hwc():
#     run_module_test(
#         SliceNet(),
#         torch.rand(1, 3, 100, 100, dtype=torch.float16),
#         input_layout=Layout.HWC,
#         output_layout=Layout.HWC,
#     )
