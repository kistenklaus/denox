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
@article ....
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

