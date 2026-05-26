from pathlib import Path

import numpy as np
from PIL import Image


def create_random_png(
    path: str | Path,
    width: int,
    height: int,
    *,
    channels: int = 3,
    seed: int = 0,
) -> Path:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if width <= 0:
        raise ValueError("width must be positive")

    if height <= 0:
        raise ValueError("height must be positive")

    if channels == 1:
        mode = "L"
        shape = (height, width)
    elif channels == 3:
        mode = "RGB"
        shape = (height, width, 3)
    elif channels == 4:
        mode = "RGBA"
        shape = (height, width, 4)
    else:
        raise ValueError("channels must be 1, 3, or 4")

    rng = np.random.default_rng(seed)

    data = rng.integers(
        0,
        256,
        size=shape,
        dtype=np.uint8,
    )

    image = Image.fromarray(
        data,
        mode=mode,
    )

    image.save(path)

    return path
