import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pydenox


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


net = TrivialConv()
net.eval()


img = Image.open("doom_pic.png").convert("RGB")

to_tensor = transforms.ToTensor()
input_tensor = to_tensor(img).unsqueeze(0)  # add batch dimension → (1, 3, H, W)

dnx = pydenox.compile_from_torch(
    net,
    input_tensor.to(torch.float16),
    input_names=("input",),
    output_names=("output",),
    input_shape=("H", "W", "C"),
    dynamic_shapes={"I": {2: torch.export.Dim.DYNAMIC, 3: torch.export.Dim.DYNAMIC}},
)

# --- 3. Run inference ---
with torch.no_grad():
    output_tensor = net(input_tensor)


# Remove batch dimension
output_tensor = output_tensor.squeeze(0)

output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

to_pil = transforms.ToPILImage()
output_img = to_pil(output_tensor)

output_img.save("output.png")
