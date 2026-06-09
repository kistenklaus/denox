Denoise Export
===========
*High performance U-Net inference, directly within your own renderer*

**denox** is a neural network compiler specifically targeting denoising U-Net 
architectures, and easy integration within existing rendering engines.
Compared to existing frameworks, denox applies more aggressive fusion, 
and produces artifacts, which can be integrated into asset pipelines,
resulting in inference within the engines-native resource and scheduling systems.

The architecture is described in detail in the accompanying paper, including benchmarking results.
```bibtex
@article{sassie2026denox, 
    author       = {Sassie, Karl and Hanika Johannes and Alber, Lucas and Dolp Reiner and Dachsbacher, Carsten},
    year         = {2026},
    title        = {Optimizing Vulkan Dispatch Schedules for Real-Time U-Net Denoising},
    volume       = {9},
    number       = {4},
    pages        = {Art.-Nr.: 53},
    journal      = {Proceedings of the ACM on Computer Graphics and Interactive Techniques},
    doi          = {10.1145/3820016},
    publisher    = {{Association for Computing Machinery (ACM)}},
}
```
#### Installing
The [latest release](https://github.com/kistenklaus/denox/releases/latest) is kept stable and is the recommended version to use.<br>
Building from source is also possible, but the main branch may occasionally fail to build.
```bash
cmake -Bbuild
cmake --build build
cmake --install build --prefix <install-dst> # optional
```

#### Documentation
- [Getting Started](docs/GettingStarted.md)
- [The DNX Format](docs/DNX.md)

