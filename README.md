# Under Construction ⚠️

#### All interfaces are open to any change!!!!

## Overview
denox is a CNN inference framework, tailored for realtime image denoising.<br>
Instead of providing a runtime, denox provides a IR of a compiled model,
which includes a set of compute dispatches including descriptors 
and SPIR-V shaders. Which when recorded on a command buffer will result in the 
CNN beeing inferred within any given runtime.

We aim to provide a simple and easy to use tool, which takes a [ONNX](https://github.com/onnx/onnx) model, 
performs optimizations like layer fusion, layout vectorization,
memory concats, selects best performing compute shaders based on benchmarking
results and finally compiles 
all of into a easy to parse IR (dnx.fbs), 
which engine developers may then parse at build time or runtime,
to translate into their engine specific api calls.
This means that engine devs will not be forced to ship large DNN libraries, within their 
products, which may decrease build-sizes significantly.

Importantly, denox is not a denoising runtime, although we have a runtime, we do not advise 
using it directly, it is only internally used for benchmarking compute shaders.
We also do not plan to optimize the memory plan, rendering engines often 
already include efficient memory aliasing logic and providing our own would 
most likely just collide with existing engine components.


#### Current Limitations
denox at this point only includes very specific ONNX operations, those include
- 3x3 convolutions 
- 2x2 max-pooling
- 2x nearest-upsampling
- ReLU
- Channel concat (skip-connections)

This set is very limited, but enough for some U-net architectures such as OIDN.

We currently also do not support texture inputs, within realtime renderers, 
networks inputs are often stored as textures, although support for textures is 
planned at this point we only accept SSBOs with a HWC or CHW layout as input.


# Getting Started
## 1. Installing the CLI
At the current state the most reliable options if you are
on a arch linux based machine is to just install denox via `makepkg`, 
on other linux distribution you will for now have to build it from source.

#### Package Managers (Recommended)
Currently we only provide a hacky solution for Arch based distributions, here you can just 
run
```bash
git clone git@github.com:kistenklaus/denox.git 
cd denox
git archive --format=tar.gz --prefix=denox-0.1.0/ -o denox-0.1.0.tar.gz HEAD
makepkg -si
cd ..
rm -rf denox 
```
to install the main branch directly. ```sudo pacman -Rs denox``` to uninstall.

#### Build from source
1. **Install dependencies**: protobuf, vulkan, spirv-tools, glslang, libpng, fmt, spdlog, vulkan-memory-allocator<br>
2. **Build project:**
```bash
git clone git@github.com:kistenklaus/denox.git
cd denox
cmake -Bbuild \
    -DCMAKE_BUILD_TYPE=Release \
    -DDENOX_UNITY_BUILD=ON \
    -DBUILD_TESTING=OFF \
    -DDENOX_SAN=OFF \
    -DDENOX_DISABLE_WARNINGS=ON  \
    -DCMAKE_INSTALL_PREFIX=~/.local \
cmake --build build -j12
```
3. **Install (optional):** ```cmake --install build``` 
only install into /usr/ or ~/.local/ if you actually know how cmake --install interoperates with 
package managers otherwise it can we quite difficult to cleanly uninstall later.
To uninstall run ```xargs rm < build/install_manifest.txt```

## 2. Exporting Pytorch to ONNX
We assume that you already have some pytorch model, which 
includes the weights. 
To export this model as a ONNX you will need the 
`onnxscript` package installed.
```python
net = Net() # <- your pytorch model!
example_input = torch.ones(1, INPUT_CHANNELS_COUNT, 5, 5, dtype=torch.float16)
program = torch.onnx.export(
    net,
    (example_input,),
    dynamic_shapes={
        "input": {2: torch.export.Dim.DYNAMIC, 3: torch.export.Dim.DYNAMIC}
    },
    input_names=["input"],
    output_names=["output"],
)
program.save("net.onnx")
```
> `onnxscript` is currently under development so this might 
change, if it doesn't work look into their latest documentation.

A couple of notes; the `dynamic_shapes` argument is 
required if your denoiser is supposed to work with 
any image size, if you do not provide this argument we will 
assume that the shape of the `example_input` is the only valid 
input. It is also very helpful to give inputs and outputs names, 
as we will later be able to refer to those by name.

## 3. Setup benchmark database
denox compilation is deliberatly structured into 
3 seperate stages, we do this because some stages 
especially benchmarking can take hours to run. 
The first step is database populate.
```
denox populate gpu.db net.onnx <COMPILER-FLAGS>
```
This step parses the `net.onnx` model and 
collects all possible compute dispatches that may
be used to infer this model, and writes them into 
the gpu.db file. If the gpu.db file already exists 
it extends the file with new dispatches.

Here are a couple of the most important flags explained:
- `--type input=f16 output=f16` overwrites the onnx models type of the `input` and `output` tensors.
    Here: `input` and `output` refer to the input / output names that we defined 
    in python when we exported the pytorch model as a ONNX model.
- `--shape input=H:W:C` this gives names to the extents of the `input` tensor,
    so in the future we can refer to the height has `H` and 
    to the width as `W`.
- `--assume H=1080 W=1920` this flag tells denox to benchmark the 
    model with `H` and `W` specialized, such that the `input` is given as a FullHD image.
- `--format input=hwc output=hwc` as ONNX and pytorch don't have any concept to distiguish between
    SSBOs and texture storages and logical tensor layouts, you have to specify the logical SSBO layout 
    yourself.
- `--use-descriptor-sets 0 1 2 3 4` tells denox to use descriptor set 0 for network inputs,
    set 1 for any network outputs, set 2 for weights, set 3 for internal readonly resources and 
    set 4 for any internal writeonly resources. If not specified all resources use set=0, but some 
    engines may require using different sets (for example set=0, could be used for globally bound resources)
- `--device=*RTX*` regex which is used to select the correct physical device, which is used to query support f
    of specific features such as cooperative matricies or what cooperative matrix shapes are available.

## 4. Benchmark database
In the second compilation step we will benchmark all dispatches within the database
for this simply run
```
denox bench gpu.db
```
Benchmarking involves measuring all dispatches within the database multiple times until
the latency converges, this can take hours as even for simple models the database generally
contains between 2-3 thousand dispatches. 
I would also recommend just leaving the PC as is, i found that doing stuff, while the 
benchmarks are running significantly increases the time it takes for the latencies to 
converge.
> There is a lot to do here so for now you will just have to be patient, in
the future we try to make this step much faster or at least ensure that the 
latencies converge much more reliable. 

> If you are on NVIDIA GPUs, i would heavily suggest locking the memory clockspeed
`nvidia-smi -lmc <max-memory-clock>`, sometimes it's also useful to lock the 
gpu-clock, but the important part is to be consistant and not run `denox bench gpu.db` once
with and once without locking the clocks.

## 5. Compile ONNX -> DNX
Lastly after we have benchmarked everything, we can now finally compile the model.
```
denox compile net.onnx -o net.dnx --database=gpu.db <COMPILER-FLAGS>
```
Compilation can now use the measurements from the `gpu.db` file and 
select the best performing dispatches (internally more or less a shortest-path algorithm).
Here you MUST pass the exact same flags as you did when you populated the database.

I would generally recomment doing all of this within a single shell script; for example:
```bash
python your_model.py
denox populate gpu.db net.onnx \
  --type input=f16 output=f16 \
  --shape input=H:W:C output=H:W:C\
  --storage input=ssbo output=ssbo \
  --format input=hwc output=hwc \
  --assume H=1080 W=1920 \
  --use-descriptor-sets 1 1 1 1 1
 
denox bench gpu.db --samples=10 --relative-error=0.015

denox compile net.onnx \
  --type input=f16 output=f16 \
  --shape input=H:W:C \
  --storage input=ssbo output=ssbo \
  --format input=hwc output=hwc \
  --assume H=1080 W=1920 \
  --use-descriptor-sets 1 1 1 1 1 \
  --db gpu.db
## (Optional) Benchmarks the final dnx model, often very useful to get an 
##            idea of how fast this model should run in your own framework.
denox bench net.dnx --spec H=1080 W=1920
```
> The first 2 compilation steps i.e. populate and bench take a couple of milliseconds 
the second time they are executed so executing them everytime is not a problem.



    



