import torch
import torch.nn.functional as F

image = torch.randn(1, 1, 32, 32)

kernel = torch.randn(1, 1, 3, 3)

output = F.conv2d(image, kernel, stride=2, padding=0)

output = F.conv2d(output, kernel, stride=1, padding=0)


print(f"Input shape: {image.shape}")
print(f"Output shape: {output.shape}")

