import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        pm = 'zeros'
        self.conv00 = nn.Conv2d(8, 8, 3, padding='same', padding_mode=pm)
        self.conv0 = nn.Conv2d(8, 8, 3, padding='same', padding_mode=pm)
        self.conv1 = nn.Conv2d(16, 16, 3, padding='same', padding_mode=pm)
        self.conv2 = nn.Conv2d(24, 8, 3, padding='same', padding_mode=pm)

    def forward(self, I):
        # NOTE YOU HAVE TO: specify CHWC8 as the input and output layout.
        x = self.conv00(I);
        a = F.relu(self.conv0(x))
        b = torch.cat([a, x], 1);
        b = F.relu(self.conv1(b))
        c = torch.cat([b,x], 1)
        return self.conv2(c)


torch.onnx.export(
        Net().eval(), 
        (torch.randn(1,8, 64,64, dtype=torch.float16),),
        "net.onnx",
        dynamo=True,
        export_params=True,
        external_data=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"I": {2 : torch.export.Dim.DYNAMIC, 3 : torch.export.Dim.DYNAMIC}},
        report=False,
)
