import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.dlpack
from torchvision import transforms
from PIL import Image
from denox import DataType, Layout, Module, Shape, Storage, TargetEnv


class TrivialPadding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, I):
        H, W = I.size(2), I.size(3)
        alignment = 2  # ensure even H/W so pool+upsample align perfectly
        H, W = I.size(2), I.size(3)
        pad_w = (alignment - (W % alignment)) % alignment
        pad_h = (alignment - (H % alignment)) % alignment
        I_aligned = F.pad(I, (100, 100, 100, 100), mode="replicate")
        x = I_aligned
        return x


class TrivialPool(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, I):
        return self.pool(I)


class TrivialUpsample(nn.Module):
    def __init__(self):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, I):
        return self.upsample(I)


class TrivialConv(nn.Module):
    def __init__(self):
        super().__init__()
        pm = "zeros"

        self.conv = nn.Conv2d(3, 3, 3, padding="same", padding_mode=pm, bias=True)

    def forward(self, I):
        return self.conv(I)


class FixedGaussianBlur(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            padding="same",
            bias=False,
        )

        # 3x3 Gaussian kernel (σ≈1)
        kernel = torch.tensor(
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ],
            dtype=torch.float16,
        )
        kernel /= kernel.sum()

        # shape (out_channels, in_channels, kH, kW)
        weight = torch.zeros((3, 3, 3, 3), dtype=torch.float16)
        for c in range(3):
            weight[c, c] = kernel

        with torch.no_grad():
            self.conv.weight.copy_(weight)

        # ensure the module runs in float16
        self.conv.to(dtype=torch.float16)

        # freeze parameters
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, I):
        # ensure input is half precision
        if I.dtype != torch.float16:
            I = I.to(dtype=torch.float16)
        return self.conv(self.conv(self.conv(I)))


net = FixedGaussianBlur()
net.eval()


img = Image.open("doom_pic.png").convert("RGB")

to_tensor = transforms.ToTensor()
input_tensor = to_tensor(img).unsqueeze(0).to(dtype=torch.float16)

program = torch.onnx.export(
    net,
    (input_tensor,),
    dynamic_shapes={"I": {2: torch.export.Dim.DYNAMIC, 3: torch.export.Dim.DYNAMIC}},
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
