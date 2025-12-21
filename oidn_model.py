# tza.py
from __future__ import annotations
import struct
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import torch
import time

import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image

from denox import DataType, Layout, Module, Shape, Storage, TargetEnv


@dataclass
class TZATensorMeta:
    name: str
    shape: Tuple[int, ...]
    layout: str  # "x" or "oihw"
    dtype_char: str  # 'f' (fp32) or 'h' (fp16)
    offset: int  # byte offset to raw tensor data


@dataclass
class TZAMeta:
    major: int
    minor: int
    tensors: Dict[str, TZATensorMeta]


def _unpack(fmt: str, data: bytes, off: int):
    """Little-endian unpack; returns (value, new_off)."""
    s = struct.Struct("<" + fmt)
    return s.unpack_from(data, off) if s.size != 1 else (
        s.unpack_from(data, off)[0],
    ), off + s.size


def parse_tza_header_and_table(path: str) -> TZAMeta:
    with open(path, "rb") as f:
        blob = f.read()
    size = len(blob)
    off = 0

    # ---- header ----
    (magic,), off = _unpack("H", blob, off)  # uint16
    if magic != 0x41D7:
        raise ValueError(f"Bad magic: 0x{magic:04X} (expected 0x41D7)")

    (major,), off = _unpack("B", blob, off)  # uint8
    (minor,), off = _unpack("B", blob, off)  # uint8
    (table_off,), off = _unpack("Q", blob, off)  # uint64

    # ---- jump to table ----
    off = table_off
    (num_tensors,), off = _unpack("I", blob, off)  # uint32

    tensors: Dict[str, TZATensorMeta] = {}
    for _ in range(num_tensors):
        # name (uint16 length + bytes)
        (name_len,), off = _unpack("H", blob, off)
        name_bytes = blob[off : off + name_len]
        off += name_len
        name = name_bytes.decode("utf-8")

        # ndims (u8)
        (ndims,), off = _unpack("B", blob, off)

        # dims (ndims * u32)
        dims = []
        for _d in range(ndims):
            (d,), off = _unpack("I", blob, off)
            dims.append(int(d))
        shape = tuple(dims)

        # layout: **ndims bytes** (e.g., b"x" or b"oihw")
        layout_bytes = blob[off : off + ndims]
        off += ndims
        layout = layout_bytes.decode("ascii")

        # dtype: one char: 'f' (fp32) or 'h' (fp16)
        (dtype_byte,), off = _unpack("c", blob, off)
        dtype_char = dtype_byte.decode("ascii")

        # tensor data offset (u64)
        (tensor_off,), off = _unpack("Q", blob, off)

        tensors[name] = TZATensorMeta(
            name=name,
            shape=shape,
            layout=layout,
            dtype_char=dtype_char,
            offset=tensor_off,
        )

    return TZAMeta(major=major, minor=minor, tensors=tensors)


def load_tza(path: str, *, copy: bool = True) -> Dict[str, np.ndarray]:
    """
    Returns {name: np.ndarray}. Uses little-endian float storage.
    """
    with open(path, "rb") as f:
        blob = f.read()
    size = len(blob)

    meta = parse_tza_header_and_table(path)
    out: Dict[str, np.ndarray] = {}

    for name, t in meta.tensors.items():
        if t.dtype_char == "f":
            dtype = np.dtype("<f4")
            item_size = 4
        elif t.dtype_char == "h":
            dtype = np.dtype("<f2")
            item_size = 2
        else:
            raise ValueError(f"Unsupported dtype '{t.dtype_char}' for tensor {name}")

        numel = int(np.prod(t.shape, dtype=np.int64))
        byte_end = t.offset + numel * item_size
        if byte_end > size:
            raise ValueError(
                f"Out-of-bounds data for tensor {name} (offset {t.offset}, size {numel * item_size})"
            )

        arr = np.frombuffer(memoryview(blob)[t.offset : byte_end], dtype=dtype).reshape(
            t.shape
        )
        out[name] = (
            np.array(arr, copy=True) if copy else arr
        )  # copy=True detaches from the file buffer

    return out


# weights_loader.py

# assumes you already have: from tza import load_tza


def _best_key_match(state_keys: List[str], tza_name: str) -> str | None:
    """Exact match first; otherwise try suffix match (handles 'module.' prefixes, etc.)."""
    if tza_name in state_keys:
        return tza_name
    candidates = [k for k in state_keys if k.endswith(tza_name)]
    if len(candidates) == 1:
        return candidates[0]
    # prefer the shortest suffix match if multiple
    if candidates:
        return sorted(candidates, key=len)[0]
    return None


def load_tza_into_model(
    model: torch.nn.Module, tza_path: str, *, verbose: bool = True
) -> Tuple[List[str], List[str]]:
    """Returns (loaded_keys, skipped_keys)."""
    tensors: Dict[str, np.ndarray] = load_tza(tza_path)

    sd = model.state_dict()
    state_keys = list(sd.keys())

    loaded, skipped = [], []
    for name, arr in tensors.items():
        k = _best_key_match(state_keys, name)
        if k is None:
            skipped.append(f"{name} (no matching param)")
            continue

        wt = torch.from_numpy(arr)  # fp16 or fp32 depending on file

        # Convert dtype to target
        target_param = sd[k]
        if target_param.dtype == torch.float32 and wt.dtype == torch.float16:
            wt = wt.float()
        elif target_param.dtype == torch.float16 and wt.dtype == torch.float32:
            wt = wt.half()

        # Shape check
        if tuple(target_param.shape) != tuple(wt.shape):
            skipped.append(
                f"{name} -> {k} (shape {tuple(wt.shape)} != {tuple(target_param.shape)})"
            )
            continue

        sd[k].copy_(wt)
        loaded.append(f"{name} -> {k}")

    model.load_state_dict(sd, strict=False)

    if verbose:
        print(f"[tza] loaded: {len(loaded)} tensors")
        for s in loaded:
            print("  ", s)
        if skipped:
            print(f"[tza] skipped: {len(skipped)}")
            for s in skipped:
                print("  ", s)

    return loaded, skipped


# 3x3 convolution module
def Conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, padding="same", padding_mode="zeros", dtype=torch.float16)


# ReLU function
def relu(x):
    return F.relu(x)


# 2x2 max pool function
def pool(x):
    return F.max_pool2d(x, 2, 2)


# 2x2 nearest-neighbor upsample function
def upsample(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


# Channel concatenation function
def concat(a, b):
    return torch.cat((a, b), 1)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, small=False):
        super(UNet, self).__init__()

        # Number of channels per layer
        ic = in_channels
        if small:
            ec1 = 32
            ec2 = 32
            ec3 = 32
            ec4 = 32
            ec5 = 32
            dc4 = 64
            dc3 = 64
            dc2a = 64
            dc2b = 32
            dc1a = 32
            dc1b = 32
        else:
            ec1 = 32
            ec2 = 48
            ec3 = 64
            ec4 = 80
            ec5 = 96
            dc4 = 112
            dc3 = 96
            dc2a = 64
            dc2b = 64
            dc1a = 64
            dc1b = 32
        oc = out_channels

        # Convolutions
        self.enc_conv0 = Conv(ic, ec1)
        self.enc_conv1 = Conv(ec1, ec1)
        self.enc_conv2 = Conv(ec1, ec2)
        self.enc_conv3 = Conv(ec2, ec3)
        self.enc_conv4 = Conv(ec3, ec4)
        self.enc_conv5a = Conv(ec4, ec5)
        self.enc_conv5b = Conv(ec5, ec5)
        self.dec_conv4a = Conv(ec5 + ec3, dc4)
        self.dec_conv4b = Conv(dc4, dc4)
        self.dec_conv3a = Conv(dc4 + ec2, dc3)
        self.dec_conv3b = Conv(dc3, dc3)
        self.dec_conv2a = Conv(dc3 + ec1, dc2a)
        self.dec_conv2b = Conv(dc2a, dc2b)
        self.dec_conv1a = Conv(dc2b + ic, dc1a)
        self.dec_conv1b = Conv(dc1a, dc1b)
        self.dec_conv0 = Conv(dc1b, oc)

        # Images must be padded to multiples of the alignment
        self.alignment = 16

    def forward(self, input):
        # Encoder
        # -------------------------------------------

        x = relu(self.enc_conv0(input))  # enc_conv0

        x = relu(self.enc_conv1(x))  # enc_conv1
        x = pool1 = pool(x)  # pool1

        x = relu(self.enc_conv2(x))  # enc_conv2
        x = pool2 = pool(x)  # pool2

        x = relu(self.enc_conv3(x))  # enc_conv3
        x = pool3 = pool(x)  # pool3

        x = relu(self.enc_conv4(x))  # enc_conv4
        x = pool(x)  # pool4

        # Bottleneck
        x = relu(self.enc_conv5a(x))  # enc_conv5a
        x = relu(self.enc_conv5b(x))  # enc_conv5b

        # Decoder
        # -------------------------------------------

        x = upsample(x)  # upsample4
        x = concat(x, pool3)  # concat4
        x = relu(self.dec_conv4a(x))  # dec_conv4a
        x = relu(self.dec_conv4b(x))  # dec_conv4b

        x = upsample(x)  # upsample3
        x = concat(x, pool2)  # concat3
        x = relu(self.dec_conv3a(x))  # dec_conv3a
        x = relu(self.dec_conv3b(x))  # dec_conv3b

        x = upsample(x)  # upsample2
        x = concat(x, pool1)  # concat2
        x = relu(self.dec_conv2a(x))  # dec_conv2a
        x = relu(self.dec_conv2b(x))  # dec_conv2b

        x = upsample(x)  # upsample1
        x = concat(x, input)  # concat1
        x = relu(self.dec_conv1a(x))  # dec_conv1a
        x = relu(self.dec_conv1b(x))  # dec_conv1b

        x = self.dec_conv0(x)  # dec_conv0

        return x


class UNetAlignment(nn.Module):
    def __init__(self, net):
        super(UNetAlignment, self).__init__()
        self.net = net

    def forward(self, input):
        alignment = self.net.alignment  # ensure even H/W so pool+upsample align perfectly
        H, W = input.size(2), input.size(3)
        pad_w = (alignment - (W % alignment)) % alignment
        pad_h = (alignment - (H % alignment)) % alignment
        aligned = F.pad(input, (0, pad_w, 0, pad_h), mode="replicate")
        output = self.net(aligned)
        return output[:,:,:H,:W]


rt_ldr = UNetAlignment(UNet(3, 3, False))
rt_ldr = rt_ldr.to(torch.float16)

load_tza_into_model(rt_ldr, "./rt_ldr.tza")

example_input = torch.ones(1, 3, 64, 64, dtype=torch.float16)

program = torch.onnx.export(
    rt_ldr,
    (example_input,),
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
#     verbose=True,
# )
# dnx.save("net.dnx")
#
#
# img = Image.open("input.png").convert("RGB")
#
# to_tensor = transforms.ToTensor()
# input_tensor: torch.Tensor = to_tensor(img).unsqueeze(0).to(dtype=torch.float16)
#
# output_tensor = torch.utils.dlpack.from_dlpack(dnx(input_tensor))
# output_tensor = output_tensor.squeeze(0)
# output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
#
# rt_ldr = rt_ldr.eval()
# device = torch.cuda.current_device()
# rt_ldr = rt_ldr.to(device=device)
# input_tensor = input_tensor.to(device=device)
#
# output_tensor_ref = rt_ldr(input_tensor)
#
# output_tensor_ref = output_tensor_ref.squeeze(0)
# output_tensor_ref = torch.clamp(output_tensor_ref, 0.0, 1.0)
#
# to_pil = transforms.ToPILImage()
#
# output_img = to_pil(output_tensor)
# output_img.save("output.png")
#
# output_ref_img = to_pil(output_tensor_ref)
# output_ref_img.save("output_ref.png")
