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
Building some basic early infrastructure:<br>
**Tensors:** We model tensors slightly more native than what pytorch or numpy does
we model them explicitly with channel count, height and width, where height. We
also seperate between Activation, Filter and Host Tensors and we encode layouts 
in a simple enumeration not in a strides array like what we see in pytorch. 
Now this might make some interoperability with torch and other frameworks slightly more
complex, but in my opinion this makes it signficantly easier to understand, at least for me.
We also introduce TensorViews, which are non owning references, that can be used to pass tensors around or 
to construct intermediate tensors from raw buffers.<br>
**Ops:** We model operations like OpConv or OpUpsample as simple structs, which encode everything that the 
we need to build a glsl implementation for such a operation. <br>
**ShaderTemplates:** Ops are realized by ShaderTemplates, those represent a small c++ wrapper glsl compute shaders,
which contain the capabilities of the shader (i.e. which ops can we implement) and given a specific operation
how do we generate the glsl code to execute those ops, this mostly boils down to which macros do we have to set here. <br>
**ShaderSources:** ShaderSources represent what a ShaderTemplate produces it's basically a instruction list on how we 
want this Shader to be executed, it also contains the shaders source code, and instructions on how to compile.<br>
**Runtime:** We also implement a small runtime, based on the merian framework, which can execute shader sources,
this is mostly for testing and we will definitely not expose this as a actual api down the line. <br>
**Testing:** We integrated the torch library for testing suites, but we are still missing a bunch of 
test suites for our shaders.<br>
**ComputeShaders:**
GLSL compute shaders for common operations in U-nets, those include:
- **ReLU**
- **Conv**
- **MaxPool**
- **NearestUpsampling**
- **CopyTransform** 

*The current implementation of all of those operations is far from optimal, but it's a great starting point.*
<br> <br>
With all of this we have some really basic infrastructure to get stuff done, but this is far from 
optimal and we should expect that most of this will have to be refactored down the line. 
###### Milestone 2: ✅
Importing of ONNX models. <br>
We first need to model our own api interface, for this we quickly created a abstraction for hypergraphs, 
and we model our api very similar to pytorches api, where we have a Model, 
which internally is just a computational hypergraph, where nodes are tensors and operations are edges. 
The Model now acts like a builder pattern, which allows us to construct such a hypergraph. 
The api is designed in such a way that we implicitly have to build the graph in topological order, 
this is not a real constrain, because when we import a model from onnx we will always get operations in 
topological order, this made a lot of the internal logic significantly easier. <br>
<br>
Importing operations from ONNX, that only operate on activation tensors like Conv, MaxPool or Upsample,
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
ConstantOfShape,Transpose and a couple more, to map the padding arithmetic to our own apis.
It's actually quite interessting how pytorch translates those operations into ONNX,
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

**Symbolic Expressions:** <br>
Designing a symbolic expression engine, was a incredibly interessting project.
For our application we needed to implement the following binary operations: **add**,**sub**,**mul**,**div**,**mod**.
Because we will only be using this for stuff like padding arithmetics we made the assumption that 
all intermediate values are unsigned integers, this assumption might be problematic later, we will see if it works out. 
Although all intermediate values must be positive, we still want to support negative values for example the 
expression X - 2, is valid. <br>
<br>
We structure the solver in 3 layers, the top most layer treats all expressions as affine combinations. 
A second layer addresses nonaffine expressions, and the last layer is a modsolver, which reduces expression in 
a given congruency class. <br>
**Affine:** Any affine expression is identified by a index into a array of affine expressions, we call those symbols.
Identities (i.e. independent variables) are affine combinations of themselfs (e.g. X = X). 
Constants are affine combinations without coefficients i.e. 4 = 4.
All affine operations (i.e. add and sub) on two values will result in another affine combinations (e.g. add(add(X,Y), 2) = X + Y + 2),
Importantly we represent the affine expression with an invariant, where the coefficients and symbols are stored in a sorted order. 
This allows us to check with a simple lookup if a identical affine expression already exists, and if it does return the symbol of this 
expression instead of creating a new symbol.
This means that we now return the identical symbol for X + Y and Y + X.
The affine layer can solve all equalities if we only work with affine operations, but for operations like mul,,div and mod, 
we can only reduce some operations within the affine layer. 
For example the expression (X + Y) * 2, can be expressed as a affine combination 2X + 2Y, 
but other multiplications like (X + 2) * Y, cannot because they are intrinsictly nonaffine. <br>
**NonAffine:** To handle nonaffine cases like (X + 2) * Y, we have to reduce the expression into a affine 
combination, we can do this by introducing a new symbol for X*Y, let's call this W, now we can express this as 
W + 2Y, which is affine. We call those new symbols nonaffine, and we store information about how we introduced them in a 
orthogonal array called the nonaffine-cache, here we apply similar caching logics as for the affine expressions, where before introducing 
a new nonaffine symbol we check if a identical already exists.
The nonaffine layer is significantly more complicated because all nonaffine symbols and affine expressions, which contain nonaffine symbols have to be 
represented in a canoncial form, otherwise we would not be able to identify equal expressions with the same symbol.
Just to give a couple of simple examples here:<br>
- (AB)C == A(BC) : This requires us to catch the case where we multiply a symbol with a nonaffine symbol, which was produced by a product we then represent this as the product of ABC.
- AB div A = B  : This requires us to detect cases of division, where the division is provable exact, so if we can represent the numerator as a multiple of the denominator we can cancel symbols.
- (AB + 2A + 4CA) div A = B + 4C + 2  : This is one of the hardest cases, because it now truely requires a canonical form for the divisor.

The canoncical representation of div, where actually quite difficult to get working.<br>
**ModSolver:**
The modsolver is the last layer, any it lazily solves all expression in a congruency class, this allows us to make strong claims 
about divisibility. For example it allows us to show that: E + (16 - (E mod 16)) mod 16, is divisible by 2, 4, 8 and 16. 
The general rules, of the modsolver are suprinsingly simple, we only have to take special care with the division operation.
If we encounter a X mod m within a modsolve of the congruency class n, then if m = n, the values is already computed and can be looked up,
otherwise we invoke another instance of the modsolver in the congruency class m. <br>
**Combining Facts:** 
The strengh of the modsolver is not only to reduce complicated modulo expressions into constants, the nonaffine layers 
can use facts from the modsolver. 
For example if we can prove that X mod 2 == 0, and we encounter (X / 2) * 2 we can conclude that this results to 
X, importantly this simplification is only possible because this division is exact. 
It is also the main driver behind what allows us to define a canonical representation for divisions,
where we can now split any affine expression of the numerator into components which are exactly divisible 
and a part which is not exactly divisible. This allows us to show identities like: <br>
- (BX + R) div B == X + R div B
- ((X - 2) / 2 + 1) * 2 == (X / 2) * 2 == X

The last equality is crucial for proving that max pooling and a following upsampling step, do not change the spatial extent.<br>

**Abusing UB:** Now if this was a right dicision iam not sure, but as stated at the very start we assume all intermediate values to be positive or zero,
explicit what this means is that we define under / overflow as UB. We also trivially define a division by zero as UB. 
This means that the div operation is allowed to show the following equality 1 / (X - 1) == 0. 
Because it will assume that X-1 != 0, otherwise it would be a division by zero. It can also assume that X - 1, does not underflow so it can conclude that 
X > 1, this means that 1 / (X - 1) == 0. This is quite a useful property, but who knowns it might come back and bite us, especially because we 
perform stride arithmetics on shapes with symbolics, when importing onnx models, and here negative strides are absolutely valid and actually required,
so with the current design we are touching a bit of self inflicted UB here, whoops =^).

###### Milestone 3: 🚧
Building a Supergraph:<br>
<br>
At this point this is just a design idea. So as a input to this stage, we get a 
computation graph, in form a Model. Here nodes represent activation tensors and edges 
represent operations. More specifically this graph is a hypergraph because we have operations 
like concat, which take multiple inputs (i.e. edges with multiple source nodes).
<br> 
The current idea, is to build a supergraph, which is again a computational hypergraph, which 
represents all possible way to compute the output. 
Edges in this graph map directly to compute shaders. 
When we build the supergraph, we directly encode optimizations like 
layout convertions, fusion and implicit concat operations.
For example a graph: <br>
```
A -Conv-> B -ReLU-> C 
```
Might be transformed into a supergraph, where the tensor B can have any layout, 
and where we include fusion edges for compute shaders which implement a Conv+Relu.
So something like this (here as a adjacency list):<br>
```
A -----DirectConv---> B 
A ------FTTConv-----> B 
A ---WinogradConv---> B 
B -Activation(ReLU)-> C 
A -----DirectConv---> C (inlined ReLU) 
A ------FTTConv-----> C (inlined ReLU) 
A ---WinogradConv---> C (inlined ReLU) 
```
In this example we ommit the layouts, so in the actual supergraph we would see node B,
duplicated as B_HWC, B_CHW, B_CHWC8.
This ofcause now means that the amount edges and nodes of the supergraph is significantly higher than the 
original computational graph. Scaling here is not on our side, but in practise i expect this 
new supergraph to not grow above a million edges, and modern computers are fast, and we do not have to perform 
any of this during runtime, so it's probably fine if it takes a couple of seconds to build this graph.
<br>
The hole idea behind building a supergraph is that we can now determine a "shortest-path" (i.e. AND-OR problem so it's actually a "shortest-subgraph"),
if we can define a heuristic which gives all edges a weight. If we assume that the herustic is correct, than 
this brute force approach will always result in the selection of the fastest possible schedule that we can 
execute. Another big architectural advantage of this approach that if we design this correctly 
it will be incredibly easy to integrate new compute shaders. We will just have to write a new compute shader 
at define a ruleset for when it is applicable, something like look at the prefix of operations in DFS order and 
if the last two operations performed on a tensor, where Conv and ReLU, apply this shader. This approach would be 
fully generic so if we find a define a shader which can implement NearestUpsampling,Concat,Conv,Relu, within a 
single dispatch and has better performance than executing those operations in series than we will 
we will be able to select this shader. 






