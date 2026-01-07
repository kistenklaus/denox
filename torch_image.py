import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.dlpack
from torchvision import transforms
from PIL import Image


def Conv(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        3,
        padding="same",
        padding_mode="zeros",
        dtype=torch.float16,
    )


class GaussianBlur(nn.Module):
    def __init__(self):
        super().__init__()

        chan = 8

        self.convgg = nn.Conv2d(
            3, chan, 3, padding="same", bias=False, dtype=torch.float16
        )

        self.convgg1 = nn.Conv2d(
            chan, chan, 3, padding="same", bias=False, dtype=torch.float16
        )

        self.convgg2 = nn.Conv2d(
            chan, 3, 3, padding="same", bias=False, dtype=torch.float16
        )
        # with torch.no_grad():
        #     for m in [self.convgg, self.convgg1, self.convgg2]:
        #         m.weight.fill_(1.0)
        #
        # self.conv0 = nn.Conv2d(
        #     3, chan, 3, padding="same", bias=False, dtype=torch.float16
        # )
        # self.conv1 = nn.Conv2d(
        #     chan, chan, 3, padding="same", bias=False, dtype=torch.float16
        # )
        #
        # self.conv2 = nn.Conv2d(
        #     chan * 2, 3, 3, padding="same", bias=False, dtype=torch.float16
        # )

        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, input):
        # H, W = input.size(2), input.size(3)
        # alignment = 4  # ensure even H/W so pool+upsample align perfectly
        # pad_w = (alignment - (W % alignment)) % alignment
        # pad_h = (alignment - (H % alignment)) % alignment
        # x = F.pad(input, (0, pad_w, 0, pad_h), mode="replicate")
        x = input

        # y = self.conv0(x)
        # z = self.conv1(y)
        #
        # x = torch.cat((z, y), 1)
        # x = self.conv2(x)

        x = self.convgg(x)
        x = self.convgg1(x)
        x = self.pool(x)
        x = self.upsample(x)
        x = self.convgg2(x)
        # x = self.convgg(x)

        return x


net = GaussianBlur()
net.eval()

assert torch.cuda.is_available(), "CUDA is NOT available"
print("CUDA device:", torch.cuda.get_device_name(0))
device = torch.device("cuda")

net = net.to(device=device)

img = Image.open("input.png").convert("RGB")

to_tensor = transforms.ToTensor()
input_tensor = to_tensor(img).unsqueeze(0).to(dtype=torch.float16, device=device)

program = torch.onnx.export(
    net,
    (input_tensor,),
    dynamic_shapes={
        "input": {2: torch.export.Dim.DYNAMIC, 3: torch.export.Dim.DYNAMIC}
    },
    input_names=["input"],
    output_names=["output"],
)
program.save("net.onnx")

# dnx = Module.compile(
#     program,
#     input_shape=Shape(H="H", W="W"),
#     summary=True,
# )
#
# output_tensor = torch.utils.dlpack.from_dlpack(dnx(input_tensor))


# --- 3. Run inference (reference) ---
with torch.no_grad():
    output_tensor_ref = net(input_tensor)


# Remove batch dimension
output_tensor_ref = output_tensor_ref.squeeze(0)
# output_tensor = output_tensor.squeeze(0)

output_tensor_ref = torch.clamp(output_tensor_ref, 0.0, 1.0)
# output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

to_pil = transforms.ToPILImage()

output_img_ref = to_pil(output_tensor_ref)
output_img_ref.save("output_ref.png")

# output_img = to_pil(output_tensor)
# output_img.save("output.png")
