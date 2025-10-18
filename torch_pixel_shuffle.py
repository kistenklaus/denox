import torch
import torch.nn as nn

class PixelShuffleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, I) -> torch.Tensor:
        return self.shuffle(I)


torch.onnx.export(
        PixelShuffleModule().eval(), 
        (torch.randn(1, 12, 5,5),),
        "pixel_shuffle.onnx",
        dynamo=True,
        export_params=True,
        external_data=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"I": {2 : torch.export.Dim.DYNAMIC, 3 : torch.export.Dim.DYNAMIC}},
        report=False,
)
