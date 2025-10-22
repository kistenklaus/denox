import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_CHANNELS_COUNT = 1

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        pm = 'zeros'
        self.enc0 = nn.Conv2d(INPUT_CHANNELS_COUNT, INPUT_CHANNELS_COUNT, 3, padding='same', padding_mode=pm, bias=False)

        self.con0 = nn.Conv2d(112, 112, 3, padding='same', padding_mode=pm)
        self.dec0 = nn.Conv2d(32, 16, 3, padding='same', padding_mode=pm)

        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')


    def forward(self, I):
        # H, W = I.size(2), I.size(3)
        # alignment = 2  # ensure even H/W so pool+upsample align perfectly
        # H, W = I.size(2), I.size(3)
        # pad_w = (alignment - (W % alignment)) % alignment
        # pad_h = (alignment - (H % alignment)) % alignment
        # I_aligned = F.pad(I, (0, pad_w, 0, pad_h), mode="replicate")
        # extr = F.relu(self.enc0(I_aligned))
        # x_128 = self.pool(extr);
        # x = self.upsample(x_128)

        return self.pool(I)

net = Net()

with torch.no_grad():
    net.enc0.weight.fill_(1.0)
    # net.enc0.bias.fill_(0.0)


torch.onnx.export(
        net,
        (torch.ones(1,INPUT_CHANNELS_COUNT, 64,64, dtype=torch.float16),),
        "net.onnx",
        dynamo=True,
        export_params=True,
        external_data=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"I": {2 : torch.export.Dim.DYNAMIC, 3 : torch.export.Dim.DYNAMIC}},
        report=False,
)
