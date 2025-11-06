import torch
from torch._C import dtype
from torch._prims_common import DeviceLikeType
import torch.nn as nn
import torch.nn.functional as F

from test import run_module_test

from denox import Layout

from torch_onnx_export import INPUT_CHANNELS_COUNT
from torch import autocast



def test_unet():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            pm = 'zeros'
            self.enc0 = nn.Conv2d(3, 32, 3, padding='same', padding_mode=pm,dtype=torch.float16)
            self.enc1 = nn.Conv2d(32, 48, 3, padding='same', padding_mode=pm,dtype=torch.float16)
            self.enc2 = nn.Conv2d(48, 64, 3, padding='same', padding_mode=pm, dtype=torch.float16)
            self.enc3 = nn.Conv2d(64, 80, 3, padding='same', padding_mode=pm, dtype=torch.float16)
            self.enc4 = nn.Conv2d(80, 112, 3, padding='same', padding_mode=pm, dtype=torch.float16)
            self.enc5 = nn.Conv2d(112, 112, 3, padding='same', padding_mode=pm, dtype=torch.float16)

            # self.extr = nn.Conv2d(32, 32, 3, padding='same', padding_mode=pm)
            
            self.dec0 = nn.Conv2d(112+112, 112, 3, padding='same', padding_mode=pm, dtype=torch.float16)
            self.con0 = nn.Conv2d(112, 112, 3, padding='same', padding_mode=pm, dtype=torch.float16)
            self.dec1 = nn.Conv2d(112+80, 80, 3, padding='same', padding_mode=pm,dtype=torch.float16)
            self.con1 = nn.Conv2d(80, 80, 3, padding='same', padding_mode=pm, dtype=torch.float16)
            self.dec2 = nn.Conv2d(80+64, 64, 3, padding='same', padding_mode=pm, dtype=torch.float16)
            self.con2 = nn.Conv2d(64, 64, 3, padding='same', padding_mode=pm, dtype=torch.float16)
            self.dec3 = nn.Conv2d(64+48, 48, 3, padding='same', padding_mode=pm, dtype=torch.float16)
            self.con3 = nn.Conv2d(48, 48, 3, padding='same', padding_mode=pm, dtype=torch.float16)
            self.dec4 = nn.Conv2d(48+32, 16, 3, padding='same', padding_mode=pm, dtype=torch.float16)
            self.con4 = nn.Conv2d(16, 16, 3, padding='same', padding_mode=pm, dtype=torch.float16)
            self.dec5 = nn.Conv2d(16+3, 12, 3, padding='same', padding_mode=pm, dtype=torch.float16)
            self.con5 = nn.Conv2d(12, 12, 3, padding='same', padding_mode=pm, dtype=torch.float16)
            
            self.pool = nn.MaxPool2d(2, 2)
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        def forward(self, input):

            alignment = 64  # ensure even H/W so pool+upsample align perfectly
            H, W = input.size(2), input.size(3)
            pad_w = (alignment - (W % alignment)) % alignment
            pad_h = (alignment - (H % alignment)) % alignment
            I_aligned = F.pad(input, (0, pad_w, 0, pad_h), mode="replicate")

            extr = F.relu(self.enc0(I_aligned))
            x_128 = self.pool(extr) 
            x_64  = self.pool(F.relu(self.enc1(x_128)))
            x_32  = self.pool(F.relu(self.enc2(x_64)))
            x_16  = self.pool(F.relu(self.enc3(x_32)))
            x_8   = self.pool(F.relu(self.enc4(x_16)))
            x_4   = self.pool(F.relu(self.enc5(x_8)))
            x     = F.relu(self.dec0(torch.cat([self.upsample(x_4), x_8],   1)))
            x     = F.relu(self.con0(x))
            x     = F.relu(self.dec1(torch.cat([self.upsample(x),   x_16],  1)))
            x     = F.relu(self.con1(x))
            x     = F.relu(self.dec2(torch.cat([self.upsample(x),   x_32],  1)))
            x     = F.relu(self.con2(x))
            x     = F.relu(self.dec3(torch.cat([self.upsample(x),   x_64],  1)))
            x     = F.relu(self.con3(x))
            x     = F.relu(self.dec4(torch.cat([self.upsample(x),   x_128], 1)))
            x     = F.relu(self.con4(x))
            x     = F.relu(self.dec5(torch.cat([self.upsample(x),   I_aligned],  1)))
            x     = F.relu(self.con5(x))

            x = x[:, :, :H, :W]
            return x
    net = Net()
    run_module_test(
        net,
        torch.rand(1, 3, 100, 100, dtype=torch.float16),
        input_layout=Layout.HWC,
        output_layout=Layout.HWC,
        # atol=0.5 # no idea if this bound is any good.
    )
