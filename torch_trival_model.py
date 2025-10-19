import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_CHANNELS_COUNT = 1

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        pm = 'zeros'
        self.conv = nn.Conv2d(1,1,3,padding="same", padding_mode=pm);

    def forward(self, I):
        x = self.conv(I);
        return x


torch.onnx.export(
        Net().eval(), 
        (torch.randn(1,INPUT_CHANNELS_COUNT, 64,64, dtype=torch.float16),),
        "net.onnx",
        dynamo=True,
        export_params=True,
        external_data=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"I": {2 : torch.export.Dim.DYNAMIC, 3 : torch.export.Dim.DYNAMIC}},
        report=False,
)
