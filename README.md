# Under Construction ⚠️
All interfaces are open to any change.<br>
We may also change the name of the project later down the line.


## Overview
vkcnn is a CNN inferance framework, tailored for realtime image denoising.<br>
Instead of providing a runtime, vkcnn provides a IR of a compiled model,
which includes a set of compute dispatches including descriptors 
and SPIR-V shaders. Which when recorded on a command buffer will result in the 
CNN beeing inferred within a given runtime.
<br>
<br>
The purpose of vkcnn is to address the the gap between research advances in image denoising
and integration of such architectures within realtime pathtracers.<br>
Training and research of image denoising architectures is often done in frameworks like 
pytorch or tensorflow, which are powered by incredibly performant backends like 
[CuDNN](https://developer.nvidia.com/cudnn), [MPS](https://developer.apple.com/documentation/metalperformanceshaders) and [DirectML](https://github.com/microsoft/DirectML),
which allow efficient training and inferrance of models on the GPU. <br>
What's problematic here is that exporting such models into a renderer is not trivial, it
requries parsing of neural network file formats like [ONNX](https://github.com/onnx/onnx), which represent the 
network as a computational graph, transforming such computational graphs into 
a efficient schedule on the GPU, is not a trivial task and it might take engine 
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




<br>
<br>
<br>
<br>

#### Development
###### Milestone 1: ✅
GLSL compute shaders for common operations in U-nets, those include:
- **ReLU**
- **Conv**
- **MaxPool**
- **NearestUpsampling**
- **CopyTransform** 

*The current implementation of all of those operations is far from optimal, but it's a great starting point.*
###### Milestone 2: ✅
Importing of ONNX models. <br>
Operations that only operate on activation tensors like Conv, MaxPool or Upsample,
where quite easy to make work, 
what made this difficult is the concat operation, 
which is common in U-nets and sometimes called a "skip-connection". <br>

If we look at this example U-net:
```python
def forward(self, x):
    x1 = F.relu(self.conv1(x))           
    x2 = self.pool(x1)           
    x2 = F.relu(self.conv2(x2))          
    x_up = self.upsample(x2)     
    x_cat = torch.cat([x1, x_up], dim=1)  
    y = self.conv3(x_cat)        
    return y
```
What happens if the input tensor ```x``` is not divisible by 2?
If ```x``` is not divisible by 2 the two inputs to the ```torch.cat``` operation 
will not have the same spatial extent. In that case the behavior of concat is unclear.
It is upto the runtime to decide how to handle this:
A runtime might crash with an exception during inferance or it might 
implicitly crop all arguments to the same size, which would mean that the 
the input ```x``` and the output ```y``` would have different spatial extent.
Both options are incredibly unattractive for realtime image denosing.

We choose the following solution to this problem.
We only allow network to be imported if we can prove that all arguments of concat have the 
same spatial extent. This now requires a symbolic expression engine to repesent the
shape of the intermediate tensors. We can now express the shapes of the intermediate tensors 
like ```x1``` and ```x_up```, by walking the computational graph and applying the correct transformations,
for example a max pool with a kernel size of (2,2) is essentially a division by 2 and so on.
With this tool we would now reject the example U-net from above because here we cannot prove that for 
any input shape the concat will see two tensors of the same spatial extent.

But if we modify the pytorch model slightly, by first padding the inputs to a multiple of 2
and then cropping the result at the very end, we can actually prove that the concat operation
will always see two arguments with the exact same spatial extent.
```python
def forward(self, x):
    H, W = x.size(2), x.size(3)
    # alignUp to multiple of 2.
    pad_w = (2 - (W % 2)) % 2
    pad_h = (2 - (H % 2)) % 2
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    x1 = F.relu(self.conv1(x))           
    x2 = self.pool(x1)           
    x2 = F.relu(self.conv2(x2))          
    x_up = self.upsample(x2)     
    x_cat = torch.cat([x1, x_up], dim=1)  
    y = self.conv3(x_cat)        
    return y[:, :, :H, :W]
```
Proving those equalities given two symbolic expressions was the first hard problem the second one was importing 
such models. Altough it might look like this in beautiful python land, the forward function is actually not really 
executed for every inferance. It is called once to build a computational model, this model expresses operations such as the 
computation of ```pad_w``` or ```pad_h```
within the same computational graph as nodes like pad or relu, we see this exact thing when we try to import this as a ONNX file.
This means that we now have to implement significantly more ONNX operations like Shape, Reshape, Sub, Mod, Gather, Unsqueeze, Cast,
ConstantOfShape,Transpose and a couple more. It's actually quite interessting how pytorch translates those operations into ONNX,
where in some places it can be incredibly verbose, which means that the graph grows signifcantly. 
For example the simple padding computation from above gets translated into 26 operations.

To summarize, we are now able to import ONNX files, by translating operations on activation tensors
directly into operation in our own IR.
Operations which are independent of the values of the activation tensors and instead only dependent on the 
shape of the tensor, we model with our symbolic expression engine, which results in a 
computational graph, where edges represent operations and nodes represent activation tensors. 
After the import, the spatial dimension of all activation tensors (nodes) is represented with symbolic expression.
This fact will later be useful for memory planning, where we can now also represent pointer offsets and peak memory usages 
with symbolic expressions.

*This was quite a interessting project with a bunch of really deep rabit holes to fall into. Although we are not able to import all onnx files, because that would 
require implementing a couple of hundred operations, we are still able to parse simple U-net architectures, which is great progess.*


###### Milestone 3: 🚧
Graph optimization.
