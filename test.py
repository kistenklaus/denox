import pydenox as dnx

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_CHANNELS_COUNT = 3

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        pm = 'zeros'
        self.enc0 = nn.Conv2d(INPUT_CHANNELS_COUNT, 32, 3, padding='same', padding_mode=pm)
        self.enc1 = nn.Conv2d(32, 48, 3, padding='same', padding_mode=pm)
        self.enc2 = nn.Conv2d(48, 64, 3, padding='same', padding_mode=pm)
        self.enc3 = nn.Conv2d(64, 80, 3, padding='same', padding_mode=pm)
        self.enc4 = nn.Conv2d(80, 112, 3, padding='same', padding_mode=pm)
        self.enc5 = nn.Conv2d(112, 112, 3, padding='same', padding_mode=pm)

        # self.extr = nn.Conv2d(32, 32, 3, padding='same', padding_mode=pm)
        
        self.dec0 = nn.Conv2d(112+112, 112, 3, padding='same', padding_mode=pm)
        self.con0 = nn.Conv2d(112, 112, 3, padding='same', padding_mode=pm)
        self.dec1 = nn.Conv2d(112+80, 80, 3, padding='same', padding_mode=pm)
        self.con1 = nn.Conv2d(80, 80, 3, padding='same', padding_mode=pm)
        self.dec2 = nn.Conv2d(80+64, 64, 3, padding='same', padding_mode=pm)
        self.con2 = nn.Conv2d(64, 64, 3, padding='same', padding_mode=pm)
        self.dec3 = nn.Conv2d(64+48, 48, 3, padding='same', padding_mode=pm)
        self.con3 = nn.Conv2d(48, 48, 3, padding='same', padding_mode=pm)
        self.dec4 = nn.Conv2d(48+32, 16, 3, padding='same', padding_mode=pm)
        self.con4 = nn.Conv2d(16, 16, 3, padding='same', padding_mode=pm)
        self.dec5 = nn.Conv2d(16+INPUT_CHANNELS_COUNT, 12, 3, padding='same', padding_mode=pm)
        self.con5 = nn.Conv2d(12, 12, 3, padding='same', padding_mode=pm)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, I):

        alignment = 64  # ensure even H/W so pool+upsample align perfectly
        H, W = I.size(2), I.size(3)
        pad_w = (alignment - (W % alignment)) % alignment
        pad_h = (alignment - (H % alignment)) % alignment
        I_aligned = F.pad(I, (0, pad_w, 0, pad_h), mode="replicate")

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

model = Net().eval()


example = torch.randn(1, 3, 64, 64, dtype=torch.float16)

prog = dnx.compile_from_torch(
    model, 
    (torch.randn(1,INPUT_CHANNELS_COUNT, 64,64, dtype=torch.float16),),
    dynamic_shapes={"I": {2 : torch.export.Dim.DYNAMIC, 3 : torch.export.Dim.DYNAMIC}},
    target_env="vulkan1.3",

    input_type="f16",
    input_layout="HWC",

    output_type="f16",
    output_layout="HWC",

    summarize=True,

)

prog.save("net.dnx")
