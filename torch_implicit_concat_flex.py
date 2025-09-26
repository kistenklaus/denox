import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        pm = 'zeros'
        self.conv00 = nn.Conv2d(8,  8, 3, padding='same', padding_mode=pm)
        self.conv0  = nn.Conv2d(8,  8, 3, padding='same', padding_mode=pm)
        self.conv1  = nn.Conv2d(16, 16, 3, padding='same', padding_mode=pm)
        # Final join after concatenating both concat outputs (40 = 16 + 24)
        self.conv3  = nn.Conv2d(40, 8, 3, padding='same', padding_mode=pm)

    def forward(self, I):
        x = self.conv00(I)                                   # (8)
        a = F.relu(self.conv0(F.relu(self.conv0(x))))        # (8)
        d = torch.cat([a, x], 1)                             # (16)  = [a | x]
        b = F.relu(self.conv1(F.relu(self.conv1(d))))        # (16)
        c = torch.cat([b, x], 1)                             # (24)  = [b | x]
        y = torch.cat([d, c], 1)                             # (40)  needs d and c simultaneously
        return self.conv3(y)                      
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
