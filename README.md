# vkcnn
Aims to be a codegeneration framework for highly 
performant CNN inferance with GLSL compute shaders.

## Design goals
Core goal is to be compatible with python DNN frameworks like 
pytorch, while also exposing a c api, 
for more realtime tasks like on the fly code generation of 
compute shaders, within game engines.
We primarily focus on U-nets as found in image denoising networks like 
OIDN.

### Code Architecture
Nothing is fixed yet, but this is just our current idea of 
what tasks actually have to be performed to achieve highly performant 
inferance.

##### Compute Graph
Our input either from python bindings or from the c api interface,
will most likely be some sort of compute graph. 
For example we might expect something like at the api interface.
```
Input(x, c)
Conv2d(3x3, stride=1, padding=1, paddingMode=zero, c=9, k=32, in=x, out=y)
Relu(in=y, out=z)
Conv2d(3x3, stride=1, padding=1, paddingMode=zero, c=32, k=32, in=z, out=w)
Output(w)
```
Importantly within this graph we will assume that the 
amount of input channels is constant, while the input image extent is 
dynamic. We also assume that the weights of Conv layers is given as a 
constant and may not change dynamically after code generation.

##### Fusion pass
The first optimization is fusion. Fusion referres to the idea of 
fusing two operations together. For example a Conv2d followed by a Relu 
can be combined into a single compute shader. 
Or even better maybe a Pool or Upsample step could be inlined 
into a previous Conv2d, which would significantly reduce the 
amount of memory reads / writes required.

> Performing the fusion pass early is a bit of a simplication, 
> as we do not perform any live benchmarking here to check if combining two
> dispatches actually leads to an improvement, we will most likely just 
> use some sort of crude heuristic here, which is obviously not going to be 
> optimal.


##### Memory planning
After the fusion pass, we can determine the livetime and relative size of 
each tensor variable (i.e. x,y,z,w). 
The livetime here is relatively simple to determine, while the 
relative size, could be complicated as we do not know the input image size.
With the livetime and sizes of the tensors computed we can now plan the 
memory layouts, where we will assume that the tensors may very well overlap 
and reuse memory regions, if their livetimes don't cross. 
During memory planning we will also have to consider the ```Concat(x,y)``` operation.
Concats could be performed as a noop if the tensors x and y are already contigous 
in memory and if their layouts match.

##### IR
Now we should have a decently minimized set of compute nodes, either by fusing or by erasing concat operations completely.
With this minimized set of operations we can now decide on how to implement the underlying operation (i.e. direct convolution vs implicit GEMM)
and the memory layouts HWC, CHW. We store the result in a small IR.
```
OpConv2dRelu(3x3, convImpl=GEMM, stride=1, padding=1, paddingMode=zero, c=9, k=32, in=x, out=z)
OpBarrier(z)
OpConv2dRelu(3x3, convImpl=GEMM, stride=1, padding=1, paddingMode=zero, c=9, k=32, in=z, out=w)
```
##### Codegen
Lastly we can go through the IR and generate code for all compute shaders, which gives us the final output,
which is now yet another IR, but a lot simpler and abstract.
```
OpCompute(glsl_src,...,in=x,out=z)
OpBarrier(z)
OpCompute(glsl_src,...,in=x,out=z)
```


### Tensor Layouts
For 3D image tensors with a image width (W) image height (H) and channels \(C\) we have a couple of different common layouts. 
CHW (channel-major) is the most useless one because it doesn't map nicely to an image tiling approach, here a 
single tile would have to load CH times W contigous channels, which for most floating point precisions and tile widths is 
going to be smaller than a 128 byte, which means that we get bad memory coalessing.
The most commonly used layout in CNNs is HWC (row-major), here each tile loads H times WC contigous values, which 
is going to be much better for memory coalessing, libraries like CUTLASS default to NHWC.
The last interessting layout is CHW4, where we essentially tile the channels as HWC images with a tile size of 4.
For example with 32bit floating point values loading 4 values (i.e. 128 bit) results in the largest 
memory access possible within a single instruction (load_vec4). Across all lanes this results in a 512byte access,
which the hardware converts into 4 128byte transactions.
For lower floating point precisions we would have to load more values at once for example for 16bit floats we would be 
using CHW8, to be able to always use vectorized load instructions. 
In addition to nicely mapping to vectorized load instructions, CHW4 had a additional advantage 
as it allows the concatenation of image tensors if both tensors have the same W and H and both have a multiple of 
4 channels. Luckily this pattern is actually quite common in U-nets for example a concat of two images with the 
same size and C1 = 64 and C2 = 112, here we can only perform the concat operation implicitly in the memory layout 
if we choose a CHW, CHW4 (f32), CHW8 (f16), CHW16 (f8) layout, with a simply HWC layout this is not possible.


### GLSL-Shadergen
Arguably the most complicated component of our design is the code generation of GLSL compute shaders,
as we have to be able to generate for for different input/output layouts, float precisions, hardware capabilities, tile sizes,
padding, strides and so on. 


##### Loading from shared memory:
One of the most important components of the shadergen is the loading from global memory tiles into 
shared memory tiles. Here it's important that we can decide on the input layout (i.e. HWC, CHW) and the 
shared memory layout (The [CUTTLASS documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/gemm_api.html#warp-level-matrix-multiply-api)
uses some pretty unique layout for implicit GEMM convolutions), as well as the floating point precision.

Initally this might seem like a trivial task, but if we aim for high performance should be aware of memory coalessing,
bank conflicts and the following optimizations applied in existing libraries.
* **Unrolling:** Especially for small tiles and small amount of input channels, unrolling the global memory accesses and index math may
  significantly increase performance, as GLSL -> SPIR-V compilers can heavily optimize such logic.
  Unrolling has the obvious downside of potentially increasing the amount of registers, but without profiling we actually don't know if that 
  is true, compilers might also just choose to reuse registers, in that case we might even reduce the amount of registers by unrolling,
  we only know the impact of unrolling after benchmarking.
* **Vectorized Loads:** Although vectorized loads do not reduce the amount of memory transactions, they reduce the amount of load instructions, which
  then reduces the amount of indexing math, which can significantly improve load performance 
  (A pretty old [article](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
  discusses this optimization, but it's also pressent in current CUTLASS vectorized loads routines, so we assume it to still be valid, but 
  we should not hope for incredible performance miracles).
  Vectorized loads are absolutely necessary if we work with lower precision floating point numbers (e.g. f16 or f8), here 
  if we would load f16s from global memory we would only issue half full transactions, which is obviously unacceptable.
* **Channel Tiling:** If the amount of input channels is large, we cannot load all input channels into shared memory, we
  instead have to load only a subrange of the channels.



