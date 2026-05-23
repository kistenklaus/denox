# DNX Format
#### Overview
The DNX FlatBuffer artifact is the serialized execution plan
emitted by *denox*.
It represents a fully materialized Vulkan-native execution schedule,
including the selected compute dispatches,
SPIR-V binaries,
tensor metadata,
resource bindings,
parameters,
and dynamic scalar expressions required for execution.
The artifact is not a high-level neural network representation.
It does not encode the original ONNX graph,
nor does it preserve abstract operators such as convolutions,
pooling,
upsampling,
activation,
or concatenation as runtime-interpreted graph nodes.
Instead,
the compiler lowers the input ONNX model
and resolves implementation choices such as operator fusion,
memory layout selection,
kernel selection,
and dispatch configuration before the artifact is written.
The resulting model is encoded as an ordered sequence
of compute dispatches.
Each dispatch references a SPIR-V shader binary,
its descriptor bindings,
push constant values,
workgroup dimensions,
resource access semantics,
and optional diagnostic metadata.
When executed in order,
the dispatch sequence performs one complete inference
without requiring graph interpretation
or runtime scheduling decisions.

The FlatBuffer also describes the device resources
used by the schedule.
Buffers describe contiguous regions of device memory,
while tensors describe how those regions are interpreted
by the generated shaders.
Model parameters are stored as tensor initializers.
Each initializer contains a binary blob
in the exact layout and scalar format expected by the selected kernel,
so that uploading parameter data can be implemented as a direct copy
without additional transformation.
The format deliberately does not prescribe a particular
renderer integration strategy.
It does not encode explicit synchronization primitives,
such as pipeline barriers,
render-graph nodes,
or resource aliasing decisions.
Instead,
each descriptor binding records the intended resource access mode,
allowing the host application to derive barriers,
resource transitions,
or render-graph dependencies according to its own scheduling system.
This keeps the artifact suitable both for engines
with sophisticated render-graph infrastructure
and for lightweight runtimes that derive synchronization directly
from the dispatch sequence.
DNX supports dynamic input resolutions
by representing size-dependent values as symbolic expressions.
Buffer sizes,
tensor offsets,
tensor extents,
dispatch dimensions,
and push constant values may depend on such expressions.
At runtime,
the scalar expressions are evaluated for the concrete input shape,
after which the resulting values can be used for resource allocation,
descriptor updates,
push constant construction,
and command recording.

#### Object Model
A DNX file is organized around a single root object,
`Model`.
The root object owns the vectors that describe
the compiled execution artifact.
Most objects in the file are either stored directly in `Model`
or are referenced by index from another object stored in `Model`.
DNX uses indices instead of nested object ownership
for most cross-references.
Unless otherwise specified,
such indices are zero-based indices
into vectors contained in `Model`.
For example,
a tensor reference is an index into `Model.tensors`,
a buffer reference is an index into `Model.buffers`,
and a shader binary reference is an index into
`Model.shader_binaries`.

The object model is divided into four main parts:
1. the resource model
2. the parameter model
3. the dispatch schedule
4. the symbolic scalar model

#### Symbolic Expressions

DNX uses symbolic expressions to represent scalar values
that may depend on runtime-provided variables (i.e. input resolution).
Typical symbolic values include buffer sizes,
tensor offsets,
tensor extents,
dispatch dimensions,
and push constant values.

Symbolic scalar values are represented by `ScalarSource`.

```fbs
table SymRef {
  sid:uint;
}

table ScalarLiteral {
  dtype:ScalarType;
  bytes:[ubyte];
}

union ScalarSource {
  literal:ScalarLiteral,
  symbolic:SymRef
}
```
Literal scalar values are stored directly in the artifact,
as a binary blob of the underyling dtype.

Symbolic values are represented by `SymRef`.
A `SymRef` refers to an entry in the value array
computed from `Model.sym_ir`.

```fbs
enum SymIROpCode:uint16 {
  NOP = 0,
  ADD = 1,
  SUB = 2,
  MUL = 3,
  DIV = 4,
  MOD = 5,
  MIN = 6,
  MAX = 7,
  LHSC = 0x8000,
  RHSC = 0x4000,
}

struct SymIROp {
  opcode:SymIROpCode;
  lhs:int64;
  rhs:int64;
}

table SymIR {
  var_count:uint16;
  ops:[SymIROp];
}
```

`SymIR` contains a linear symbolic expression program.
The program is evaluated into a temporary array of `int64_t` values.
Let `values` be an array with size 
`sym_ir.var_count + sym_ir.ops.size()`
The first `sym_ir.var_count` entries contain the runtime-provided
symbolic variables.
The remaining entries contain the results of the operations
stored in `sym_ir.ops`.

For operation `i`,
the result is written to
`values[sym_ir.var_count + i]`.
The operations in `sym_ir.ops` are stored in evaluation order.
An operation may reference symbolic variables
or previously computed operation results.
After the symbolic expression program has been evaluated,
`SymRef.sid` is interpreted as an index into `values`.

The opcode of a `SymIROp` contains both the base operation
and operand interpretation flags.
The base operation is obtained by removing the operand flags.

```text
base_opcode = opcode & ~(LHSC | RHSC)
```
If `LHSC` is set,
`lhs` is interpreted as an immediate constant.
Otherwise,
`lhs` is interpreted as an index into `values`.
If `RHSC` is set,
`rhs` is interpreted as an immediate constant.
Otherwise,
`rhs` is interpreted as an index into `values`.

The following base operations are defined:

| Opcode | Operation |
|---|---|
| `ADD` | addition |
| `SUB` | subtraction |
| `MUL` | multiplication |
| `DIV` | integer flooring division |
| `MOD` | euclidean modulo |
| `MIN` | minimum |
| `MAX` | maximum |

`MOD` denotes euclidean modulo.
For operands `a` and `b`,
with `b > 0`,
the result is defined as
`a - b * floor(a / b)`.
The result is in the range
`0 <= result < b`
This is not necessarily identical to the host language `%` operator
for signed integer operands.

Conceptually,
a runtime evaluates `SymIR` as follows:
```cpp
std::vector<int64_t> values(
    sym_ir.var_count + sym_ir.ops.size());

for (size_t i = 0; i < sym_ir.var_count; ++i) {
    values[i] = variables[i];
}

for (size_t i = 0; i < sym_ir.ops.size(); ++i) {
    const SymIROp op = sym_ir.ops[i];

    const auto base =
        op.opcode & ~(LHSC | RHSC);

    const bool lhs_is_constant =
        (op.opcode & LHSC) != 0;

    const bool rhs_is_constant =
        (op.opcode & RHSC) != 0;

    const int64_t lhs =
        lhs_is_constant
            ? op.lhs
            : values[static_cast<size_t>(op.lhs)];

    const int64_t rhs =
        rhs_is_constant
            ? op.rhs
            : values[static_cast<size_t>(op.rhs)];

    values[sym_ir.var_count + i] =
        evaluate(base, lhs, rhs);
}
```

A symbolic scalar source is resolved by indexing
the evaluated value array.
```cpp
int64_t resolve_sym_ref(
    SymRef ref,
    std::span<const int64_t> values)
{
    return values[ref.sid];
}
```

##### Named symbolic values
Symbolic values may optionally be assigned names.
```fbs
table ValueName {
  name:string;
  value:ScalarSource;
}
```
`ValueName` associates a human-readable name with a scalar value.
The named value may be either a literal value
or a symbolic reference into the evaluated `SymIR` value array.
Named values are primarily intended for reflection,
diagnostics,
and host-side binding of runtime variables.

For example,
a model may expose symbolic input dimensions using names such as
`"W"` and `"H"`.
This is especially useful for symbolic variables,
because the order of variables in `SymIR` is not part of the stable
user-facing interface
and may change between artifacts.

A host application should therefore not rely on a fixed variable order
when assigning runtime-provided values.
Instead,
it should inspect `Model.value_names`,
find the entries corresponding to the required variable names,
and use the associated `ScalarSource` to determine which symbolic value
is being named.

If a named value refers to a symbolic variable,
then its `SymRef.sid` is smaller than `sym_ir.var_count`.
Such entries can be used to bind runtime-provided values
before evaluating `sym_ir.ops`.

For example,
an artifact may contain the following conceptual bindings:
```text
"W" -> values[0]
"H" -> values[1]
```
The runtime may then initialize the symbolic variable array as follows:
```cpp
values[0] = input_width;
values[1] = input_height;
```
After the symbolic expression program has been evaluated,
a named value is resolved in the same way as any other `ScalarSource`.

#### Buffers & Tensors
DNX separates backing storage from tensor interpretation.
A `Buffer` describes a contiguous region of device memory.
A `Tensor` describes how a region of storage is interpreted
by generated shaders.

```fbs
table Buffer {
  size:ScalarSource;
  alignment:ushort;
}

table Tensor {
  buffer:uint;
  offset:ScalarSource;
  size:ScalarSource;
  info:TensorInfo;
}
```
`Model.buffers` contains the logical buffers required by the model.
`Model.tensors` contains the logical tensors used by the dispatch schedule.
A tensor references its backing buffer by index.
The `buffer` field of `Tensor` is a zero-based index into
`Model.buffers`.

The `offset` field specifies the byte offset of the tensor data
inside the referenced buffer.
The `size` field specifies the size of the tensor storage region
in bytes.
Both `offset` and `size` are `ScalarSource` values.
They may therefore be either literal values
or symbolic expressions depending on the input shape.

The `size` field of `Buffer` specifies the required buffer size
in bytes.
The `alignment` field specifies the required alignment
of the buffer allocation.
A runtime shall allocate at least `size` bytes
for each buffer
and satisfy the specified alignment requirement.

Multiple tensors may refer to the same buffer.
In that case,
their `offset` and `size` fields describe distinct,
overlapping,
or otherwise implementation-defined regions
within the same allocation.

The DNX format does not prescribe a memory aliasing strategy.
The runtime is responsible for emitting tensor ranges
that are valid for the compiled dispatch schedule.

The host application may still map buffers
to engine-specific resources,
perform aliasing,
or integrate them into a renderer-specific memory allocator,
provided that the observable behavior of the dispatch schedule
is preserved.

Tensor metadata is stored in `TensorInfo`.
```fbs
enum TensorFormat:ushort {
  UNKNOWN,
  SSBO_HWC,
  SSBO_CHW,
  SSBO_CHWC8,
  TEX_RGBA,
  TEX_RGB,
  TEX_RG,
  TEX_R
}

enum TensorStorage:ushort {
  StorageBuffer,
  StorageImage,
  SampledStorageImage,
}

table TensorInfo {
  width:ScalarSource;
  height:ScalarSource;
  channels:ScalarSource;
  format:TensorFormat;
  storage:TensorStorage;
  type:ScalarType;
  name:string;
}
```
`TensorInfo` describes the logical shape,
physical layout,
storage class,
element type,
and optional name of a tensor.

The `width`,
`height`,
and `channels` fields describe the logical tensor extent.

The `format` field describes the physical tensor layout
expected by the generated shaders.

The `storage` field describes the kind of Vulkan resource
used to store or access the tensor.

The `type` field specifies the scalar element type.

The `name` field is an optional human-readable name
intended for diagnostics,
reflection,
and integration code.

The following tensor formats are defined:

| Format | Description |
|---|---|
| `UNKNOWN` | Layout is unspecified or implementation-defined. |
| `SSBO_HWC` | Buffer tensor with channel-major layout. |
| `SSBO_CHW` | Buffer tensor with channel-minor layout. |
| `SSBO_CHWC8` | Buffer tensor with vectroized CHW layout. |
| `TEX_RGBA` | Image tensor with four components. |
| `TEX_RGB` | Image tensor with three components. |
| `TEX_RG` | Image tensor with two components. |
| `TEX_R` | Image tensor with one component. |
> Textures are not supported at the time of writing this documentation.

The following tensor storage classes are defined:

| Storage | Description |
|---|---|
| `StorageBuffer` | Tensor is accessed as a storage buffer. |
| `StorageImage` | Tensor is accessed as a storage image. |
| `SampledStorageImage` | Read with a sampler and write as a storage image. |
> StorageImage and SampledStorageImage are not supported at the time of writing this documentation.

A tensor does not own its backing storage.
It is a view into storage described elsewhere in the model
or provided by the host application.

#### Tensor Initializers
Tensor initializers store constant tensor data embedded
in the DNX artifact.
Initializers are used for model parameters,
such as weights,
biases,
and other compile-time constant tensors.

```fbs
table TensorInitializer {
  tensor:uint (key);
  data:[ubyte];
}
```

`Model.initializers` contains the list of tensor initializers
stored in the artifact.

Each initializer references exactly one tensor.

The referenced `tensor` is never written to by any dispatch.

The `tensor` field is a zero-based index into `Model.tensors`.

The `data` field contains the raw binary contents
used to initialize the referenced tensor.

The size of `data` must be smaller than the size of the referenced `tensor`.

The binary data is stored in the physical layout
and scalar format expected by the selected shader implementation.
A runtime should therefore treat initializer data
as an opaque byte sequence.
No layout conversion,
type conversion,
or reordering is required when uploading an initializer,
unless such conversion is introduced by the host application itself.

Before a dispatch reads from a tensor
that has an associated initializer,
the initializer data must be available
in the corresponding device resource.

The DNX format does not prescribe how initialized tensors
are managed by the host application.
A runtime may upload initializer data once during model loading,
stream it into device memory before execution,
or integrate it into an engine-specific asset system.

#### Dispatches
A `ComputeDispatch` describes one Vulkan compute dispatch
in the compiled execution schedule.

It specifies the shader binary,
entry point,
workgroup dimensions,
descriptor bindings,
push constant data,
resource access information,
and optional diagnostic metadata required to record the dispatch.

```fbs
table ComputeDispatch {
  binary_id:uint32;
  workgroup_count_x:ScalarSource;
  workgroup_count_y:ScalarSource;
  workgroup_count_z:ScalarSource;
  entry_point:string;
  bindings:[DescriptorSetBinding];
  push_constant:PushConstant;

  info:DispatchInfo;
  requirements:ComputeDispatchRequirements;
}
```
`Model.dispatches` contains the ordered list of compute dispatches
that form the compiled model schedule.

Each `ComputeDispatch` corresponds conceptually to binding
one compute pipeline,
binding its descriptor sets,
writing its push constants,
and issuing one `vkCmdDispatch` command.

The `binary_id` field is a zero-based index into
`Model.shader_binaries`.

The `entry_point` field names the compute entry point
inside the selected SPIR-V module.

The three workgroup count fields specify the dispatch dimensions.
They are represented as `ScalarSource` values,
so they may be either literal values
or symbolic expressions depending on runtime variables
such as the input resolution.

After resolving these values,
they correspond to the arguments passed to `vkCmdDispatch`.

```cpp
vkCmdDispatch(
    command_buffer,
    workgroup_count_x,
    workgroup_count_y,
    workgroup_count_z);
```

A dispatch may use descriptor bindings
to access tensors.

```fbs
enum Access:ubyte {
  ReadOnly = 0,
  WriteOnly = 1,
  ReadWrite = 2
}

table DescriptorBinding {
  binding:ushort;
  access:Access;
  tensor:uint;
}

table DescriptorSetBinding {
  set:ushort;
  bindings:[DescriptorBinding];
}
```

`bindings` describes the descriptor sets
and descriptor bindings required by the dispatch.

Each `DescriptorSetBinding` corresponds to one descriptor set number.

Each `DescriptorBinding` inside it corresponds to one binding
within that set.

The `tensor` field is a zero-based index into `Model.tensors`.
The referenced tensor determines the descriptor type
that should be bound at the specified descriptor set and binding.

The `access` field describes how the shader accesses the tensor.

| Access | Meaning |
|---|---|
| `ReadOnly` | The dispatch reads from the tensor. |
| `WriteOnly` | The dispatch writes to the tensor. |
| `ReadWrite` | The dispatch may both read from and write to the tensor. |

The access mode is metadata for the host application.
It may be used to derive pipeline barriers,
resource state transitions,
or render-graph dependencies.
It does not by itself encode a Vulkan synchronization command.

A dispatch may also define push constant data.
```fbs
table PushConstantField {
  dtype:ScalarType;
  offset:ushort;
  source:ScalarSource;
}

table PushConstant {
  size:ushort;
  fields:[PushConstantField];
}
```

`push_constant` describes the push constant payload
used by the dispatch.

The `size` field specifies the total number of bytes
in the push constant range.

Each `PushConstantField` describes one scalar value
written into this range.

The `offset` field specifies the byte offset
within the push constant range.

The `dtype` field specifies the scalar type
used to encode the value.

The `source` field specifies the scalar value
that should be written.

It may be a literal value
or a symbolic value resolved from `Model.sym_ir`.

Conceptually,
a runtime constructs a temporary push constant byte buffer,
writes each field into the buffer at its declared offset,
and passes the resulting byte range to `vkCmdPushConstants`.

```cpp
std::vector<std::byte> push_constant_data(
    dispatch.push_constant.size);

for (const PushConstantField& field :
     dispatch.push_constant.fields) {
    const auto value =
        resolve_scalar(field.source);

    write_scalar(
        push_constant_data.data() + field.offset,
        field.dtype,
        value);
}

vkCmdPushConstants(
    command_buffer,
    pipeline_layout,
    VK_SHADER_STAGE_COMPUTE_BIT,
    0,
    push_constant_data.size(),
    push_constant_data.data());
```

The dispatch may also contain additional requirements
on the execution environment.

```fbs
table ComputeDispatchRequirements {
  fixed_subgroup_size:uint32;
}
```

`fixed_subgroup_size` specifies the subgroup size required
by this dispatch.

A value of `0` indicates that the dispatch does not require
a fixed subgroup size.

If `fixed_subgroup_size` is nonzero,
the host application must ensure that the compute pipeline
is created and executed with the specified subgroup size (VK_EXT_subgroup_size_control),
or reject the dispatch.

Diagnostic metadata is stored in `DispatchInfo`.
```fbs
table DispatchInfo {
  name:string;
  src_path:string;
  memory_reads:ScalarSource;
  memory_writes:ScalarSource;
  flops:ScalarSource;
}
```

`DispatchInfo` is not required to record the Vulkan command.

A `ComputeDispatch` can therefore be viewed as a compact,
serializable description of the following Vulkan-level operation:

```cpp
// Conceptual mapping only.

const ShaderBinary& shader =
    model.shader_binaries[dispatch.binary_id];

VkShaderModule module =
    create_shader_module(shader.spirv);

VkPipeline pipeline =
    create_compute_pipeline(
        module,
        dispatch.entry_point,
        dispatch.requirements);

bind_descriptor_sets(
    command_buffer,
    pipeline_layout,
    dispatch.bindings);

push_constants(
    command_buffer,
    pipeline_layout,
    dispatch.push_constant);

vkCmdDispatch(
    command_buffer,
    resolve(dispatch.workgroup_count_x),
    resolve(dispatch.workgroup_count_y),
    resolve(dispatch.workgroup_count_z));
```

The exact construction of Vulkan objects,
descriptor sets,
pipeline layouts,
and synchronization primitives is outside the scope
of `ComputeDispatch`.

Those details are intentionally left to the host application
or renderer integration layer.

#### Execution Model
The DNX execution model is intentionally simple.

Executing a DNX model consists of evaluating dynamic scalar values,
preparing the required resources,
binding the resources described by each dispatch,
and recording the dispatches in the order stored in `Model.dispatches`.

A runtime conceptually performs the following steps:

```text
load Model
create shader modules and compute pipelines
bind runtime-provided symbolic variables
evaluate Model.sym_ir
resolve buffer sizes, tensor ranges, and dispatch dimensions
update descriptors
allocate resources
upload tensor initializers
for each dispatch in Model.dispatches:
    bind the selected compute pipeline
    bind descriptors
    bind push constants
    derive required synchronization from access semantic
    issue vkCmdDispatch
```

The exact mapping to renderer infrastructure is intentially implementation-defined,
and the order above can be changed to the hearts content.

##### Synchronization
DNX does not encode explicit synchronization commands.
It does not contain Vulkan pipeline barriers,
events,
semaphores,
or render-graph dependency edges.
Instead,
each descriptor binding declares the access mode
with which the dispatch uses the referenced tensor.

The host application may use this information,
together with the order of `Model.dispatches`,
to derive synchronization.

For example,
a write to a tensor followed by a read of the same tensor
requires appropriate synchronization before the read dispatch executes.
A renderer with an existing render graph may translate the dispatch list
and access metadata into graph nodes and resource edges.
A lightweight runtime may instead derive barriers directly
while iterating over the dispatch list.

##### Completion and outputs
After all dispatches have executed,
the tensors listed in `Model.outputs`
contain the externally visible model results.

The format does not prescribe how these results are consumed.

They may remain in renderer-owned device resources,
be used as inputs to later rendering passes,
or be copied back to host-visible memory.
