import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.dlpack
from torchvision import transforms
from PIL import Image
from denox import DataType, Layout, Module, Shape, Storage, TargetEnv


class GaussianBlur(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ],
            dtype=torch.float16,
        )
        kernel /= kernel.sum()
        weight = torch.zeros((3, 3, 3, 3), dtype=torch.float16)
        weight[0, 0] = kernel
        weight[1, 1] = kernel
        weight[2, 2] = kernel

        self.conv = nn.Conv2d(3, 3, 3, padding="same", bias=False, dtype=torch.float16)
        with torch.no_grad():
            self.conv.weight.copy_(weight)

    def forward(self, x):
        for _ in range(100):
            x = self.conv(x)
        return x


net = GaussianBlur()
net.eval()


img = Image.open("doom_pic.png").convert("RGB")

to_tensor = transforms.ToTensor()
input_tensor = to_tensor(img).unsqueeze(0).to(dtype=torch.float16)

program = torch.onnx.export(
    net,
    (input_tensor,),
    dynamic_shapes={"x": {2: torch.export.Dim.DYNAMIC, 3: torch.export.Dim.DYNAMIC}},
    input_names=["input"],
    output_names=["output"],
)

dnx = Module.compile(
    program,
    input_shape=Shape(H="H", W="W"),
    summary=True,
)

output_tensor = torch.utils.dlpack.from_dlpack(dnx(input_tensor))


# --- 3. Run inference (reference) ---
with torch.no_grad():
    output_tensor_ref = net(input_tensor)


# Remove batch dimension
output_tensor_ref = output_tensor_ref.squeeze(0)
output_tensor = output_tensor.squeeze(0)

output_tensor_ref = torch.clamp(output_tensor_ref, 0.0, 1.0)
output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

to_pil = transforms.ToPILImage()

output_img_ref = to_pil(output_tensor_ref)
output_img_ref.save("output_ref.png")

output_img = to_pil(output_tensor)
output_img.save("output.png")
