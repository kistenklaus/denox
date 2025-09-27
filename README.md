# Under Construction ⚠️


#### All interfaces are open to any change!!!!

## Overview
denox is a CNN inferance framework, tailored for realtime image denoising.<br>
Instead of providing a runtime, denox provides a IR of a compiled model,
which includes a set of compute dispatches including descriptors 
and SPIR-V shaders. Which when recorded on a command buffer will result in the 
CNN beeing inferred within a given runtime.
<br>
<br>
The purpose of denox is to address the the gap between research advances in image denoising
and integration of such architectures within realtime pathtracers.<br>
Training and research of image denoising architectures is often done in frameworks like 
pytorch or tensorflow, which are powered by incredibly performant backends like 
[CuDNN](https://developer.nvidia.com/cudnn), [MPS](https://developer.apple.com/documentation/metalperformanceshaders) and [DirectML](https://github.com/microsoft/DirectML),
which allow efficient training and inferrance of models on the GPU. <br>
What's problematic here is that exporting such models into a renderer is not trivial, it
requries parsing of neural network file formats like [ONNX](https://github.com/onnx/onnx), which represent the 
network as a computational graph, transforming such computational graphs into 
a efficient schedule on the GPU, is not a trivial task and 
manually implementing a neural network might take engine 
developer years to reach acceptable performance.
<br>
<br>
We aim to provide a simple and easy to use tool, which takes a [ONNX](https://github.com/onnx/onnx) model, 
performs optimizations like layer fusion, layout vectorization,
implicit concats and possibly more.
We will also precompute a memory plan, which determines peak memory 
usage and device address offsets of the individual pointers, while 
considering the lifetimes of tensors and reusing memory regions no 
longer in need. <br>
All of this will be packed into a easy to parse IR (possibly a protobuf), which engine 
developers may then parse at build time, to translate into their engine specific api calls.
This means that engine devs will not be forced to ship large DNN libraries, within their 
products, which may decrease build-sizes significantly.


## Getting Started
If you interessted in running the current version:
Checkout the Github Actions tab and look for a commit, where the 
CI passed.

##### 1. Install dependencies (Ubuntu)
```
sudo apt-get install -y protobuf-compiler libprotobuf-dev libfmt-dev libvulkan1 vulkan-tools libvulkan-dev libshaderc-dev glslc glslang-tools
```
If you are on another linux distro, just make sure that ```protobuf``` and```fmt```is installed,
but on most systems they should be installed anyway, as part of other packages.

##### 2. Generate example onnx file.
```
pip install torch
pip install onnxscript
python torch_onnx_example.py 
```
##### 3. Build & Run project
```
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release
cmake --build build -j16 
./build/denox
```
Make sure that you run the executable from the directory where the 
example onnx file is (i.e. ```net.onnx```)

##### 4. Running the test suite.
```
cmake --build build -j16 --target=denox_test
./build/denox_test
```



