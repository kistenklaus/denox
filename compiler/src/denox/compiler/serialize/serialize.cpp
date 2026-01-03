#include "denox/compiler/serialize/serialize.hpp"
#include "denox/algorithm/align_up.hpp"
#include "denox/common/Access.hpp"
#include "denox/common/TensorDataType.hpp"
#include "denox/compiler/compile_shaders/SpvDispatch.hpp"
#include "denox/compiler/placement/TensorInitalizer.hpp"
#include "denox/device_info/query/query_resource_limits.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/logging.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/memory/dtype/f16.hpp"
#include "denox/memory/dtype/f32.hpp"
#include "denox/memory/dtype/f64.hpp"
#include "denox/spirv/SpirvBinary.hpp"
#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/vector.h"
#include <concepts>
#include <dnx.h>
#include <limits>

namespace denox::compiler {

static denox::dnx::Version serialize_version([[maybe_unused]] unsigned int version) {
  return denox::dnx::Version_DNX_VERSION_1_0;
}

static flatbuffers::Offset<flatbuffers::Vector<uint32_t>>
serialize_required_features(flatbuffers::FlatBufferBuilder &fbb) {
  memory::vector<uint32_t> requiredFeatures;
  requiredFeatures.push_back(denox::dnx::RuntimeFeature_CooperativeMatrix);
  return fbb.CreateVector<uint32_t>(requiredFeatures.data(),
                                    requiredFeatures.size());
}

static flatbuffers::Offset<denox::dnx::ModelInfo>
serialize_model_info(flatbuffers::FlatBufferBuilder &fbb,
                     const ModelMeta &meta) {

  flatbuffers::Offset<flatbuffers::String> producer = 0;
  flatbuffers::Offset<flatbuffers::String> producer_version = 0;
  flatbuffers::Offset<flatbuffers::String> model_version = 0;

  if (meta.producerName) {
    producer = fbb.CreateString(*meta.producerName);
  }
  if (meta.producerVersion) {
    producer_version = fbb.CreateString(*meta.producerVersion);
  }
  if (meta.modelVersion) {
    model_version = fbb.CreateString(*meta.modelVersion);
  }
  return dnx::CreateModelInfo(fbb, producer, producer_version, model_version);
}

static denox::dnx::ScalarType serialize_type(memory::Dtype type) {
  switch (type.kind()) {
  case memory::DtypeKind::F16:
    return denox::dnx::ScalarType_F16;
  case memory::DtypeKind::F32:
    return denox::dnx::ScalarType_F32;
  case memory::DtypeKind::F64:
    return denox::dnx::ScalarType_F64;
  case memory::DtypeKind::U32:
    return denox::dnx::ScalarType_U32;
  case memory::DtypeKind::I32:
    return denox::dnx::ScalarType_I32;
  case memory::DtypeKind::U64:
    return denox::dnx::ScalarType_U64;
  case memory::DtypeKind::I64:
    return denox::dnx::ScalarType_I64;
  }
  diag::unreachable();
}

static denox::dnx::ScalarType serialize_type(TensorDataType type) {
  switch (type) {
  case TensorDataType::Auto:
    DENOX_ERROR("Failed to serialize: found auto type in final artefact");
    diag::invalid_state();
  case TensorDataType::Float16:
    return denox::dnx::ScalarType_F16;
  case TensorDataType::Float32:
    return denox::dnx::ScalarType_F32;
  case TensorDataType::Float64:
    return denox::dnx::ScalarType_F64;
  }
  diag::unreachable();
}

static denox::dnx::TensorFormat serialize_tensor_format(TensorFormat format) {
  switch (format) {
  case TensorFormat::Optimal:
    return denox::dnx::TensorFormat::TensorFormat_UNKNOWN;
  case TensorFormat::SSBO_HWC:
    return denox::dnx::TensorFormat_SSBO_HWC;
  case TensorFormat::SSBO_CHW:
    return denox::dnx::TensorFormat_SSBO_CHW;
  case TensorFormat::SSBO_CHWC8:
    return denox::dnx::TensorFormat_SSBO_CHWC8;
  case TensorFormat::TEX_RGBA:
    return denox::dnx::TensorFormat_TEX_RGBA;
  case TensorFormat::TEX_RGB:
    return denox::dnx::TensorFormat_TEX_RGB;
  case TensorFormat::TEX_RG:
    return denox::dnx::TensorFormat_TEX_RG;
  case TensorFormat::TEX_R:
    return denox::dnx::TensorFormat_TEX_R;
  }
  diag::unreachable();
}

static denox::dnx::TensorStorage
serialize_tensor_storage(TensorStorage storage) {
  switch (storage) {
  case TensorStorage::Optimal:
    DENOX_ERROR("Failed to serialize: found optimal storage in final artefact");
    diag::invalid_state();
  case TensorStorage::StorageBuffer:
    return denox::dnx::TensorStorage_StorageBuffer;
  case TensorStorage::StorageImage:
    return denox::dnx::TensorStorage_StorageImage;
  case TensorStorage::SampledStorageImage:
    return denox::dnx::TensorStorage_SampledStorageImage;
  }
  diag::unreachable();
}

static denox::dnx::Access serialize_access(denox::Access access) {
  switch (access) {
  case Access::ReadOnly:
    return denox::dnx::Access_ReadOnly;
  case Access::WriteOnly:
    return denox::dnx::Access_WriteOnly;
  case Access::ReadWrite:
    return denox::dnx::Access_ReadWrite;
  }
  diag::unreachable();
}

template <typename T>
static flatbuffers::Offset<denox::dnx::ScalarLiteral>
serialize_literal(flatbuffers::FlatBufferBuilder &fbb, T value,
                  denox::dnx::ScalarType type) {
  memory::vector<uint8_t> bytes;
  switch (type) {
  case dnx::ScalarType_I16: {
    bytes.resize(2);
    int16_t x{static_cast<int16_t>(value)};
    std::memcpy(bytes.data(), &x, 2);
    break;
  }
  case dnx::ScalarType_U16: {
    bytes.resize(2);
    uint16_t x{static_cast<uint16_t>(value)};
    std::memcpy(bytes.data(), &x, 2);
    break;
  }
  case dnx::ScalarType_I32: {
    bytes.resize(4);
    int32_t x{static_cast<int32_t>(value)};
    std::memcpy(bytes.data(), &x, 4);
    break;
  }
  case dnx::ScalarType_U32: {
    bytes.resize(4);
    uint32_t x{static_cast<uint32_t>(value)};
    std::memcpy(bytes.data(), &x, 4);
    break;
  }
  case dnx::ScalarType_I64: {
    bytes.resize(8);
    int64_t x{static_cast<int64_t>(value)};
    std::memcpy(bytes.data(), &x, 4);
    break;
  }
  case dnx::ScalarType_U64: {
    bytes.resize(8);
    uint64_t x{static_cast<uint64_t>(value)};
    std::memcpy(bytes.data(), &x, 8);
    break;
  }
  case dnx::ScalarType_F16:
  case dnx::ScalarType_F32:
  case dnx::ScalarType_F64:
    diag::unreachable();
  }
  auto boffset = fbb.CreateVector<uint8_t>(bytes.data(), bytes.size());
  return dnx::CreateScalarLiteral(fbb, type, boffset);
}

static flatbuffers::Offset<denox::dnx::SymRef>
serialize_symbol(flatbuffers::FlatBufferBuilder &fbb, Sym::symbol symbol) {
  return dnx::CreateSymRef(fbb, static_cast<uint32_t>(symbol));
}

static std::pair<denox::dnx::ScalarSource, flatbuffers::Offset<void>>
serialize_scalar(flatbuffers::FlatBufferBuilder &fbb, Sym sym,
                 memory::Dtype type) {
  if (sym.isSymbolic()) {
    return std::make_pair(denox::dnx::ScalarSource_symbolic,
                          serialize_symbol(fbb, sym.sym()).Union());
  } else {
    auto dtype = serialize_type(type);
    return std::make_pair(
        denox::dnx::ScalarSource_literal,
        serialize_literal(fbb, sym.constant(), dtype).Union());
  }
}

static flatbuffers::Offset<denox::dnx::TensorInfo>
serialize_tensor_info(flatbuffers::FlatBufferBuilder &fbb,
                      const TensorInfo &info) {

  auto [width_type, width] =
      serialize_scalar(fbb, info.width, memory::Dtype::U32);
  auto [height_type, height] =
      serialize_scalar(fbb, info.height, memory::Dtype::U32);
  auto [channel_type, channels] =
      serialize_scalar(fbb, info.channels, memory::Dtype::U32);

  auto format = serialize_tensor_format(info.format);
  auto storage = serialize_tensor_storage(info.storage);
  auto type = serialize_type(info.type);

  flatbuffers::Offset<flatbuffers::String> name = 0;
  if (info.name) {
    name = fbb.CreateString(*info.name);
  }

  return dnx::CreateTensorInfo(fbb, width_type, width, //
                               height_type, height,    //
                               channel_type, channels, //
                               format, storage, type, name);
}

static flatbuffers::Offset<denox::dnx::Tensor>
serialize_tensor(flatbuffers::FlatBufferBuilder &fbb,
                 const TensorView &tensor) {
  auto [offset_type, offset] =
      serialize_scalar(fbb, tensor.offset, memory::Dtype::U32);
  auto [size_type, size] =
      serialize_scalar(fbb, tensor.offset, memory::Dtype::U32);
  auto info = serialize_tensor_info(fbb, tensor.info);
  return dnx::CreateTensor(fbb,
                           static_cast<uint32_t>(tensor.buffer), //
                           offset_type, offset,                  //
                           size_type, size, info);
}

static flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<denox::dnx::Tensor>>>
serialize_tensors(flatbuffers::FlatBufferBuilder &fbb,
                  const memory::span<const TensorView> tensors) {
  memory::vector<flatbuffers::Offset<denox::dnx::Tensor>> offsets(
      tensors.size());
  for (size_t t = 0; t < tensors.size(); ++t) {
    offsets[t] = serialize_tensor(fbb, tensors[t]);
  }
  return fbb.CreateVector<denox::dnx::Tensor>(offsets.data(), offsets.size());
}

static flatbuffers::Offset<denox::dnx::TensorInitializer>
serialize_initializer(flatbuffers::FlatBufferBuilder &fbb,
                      const TensorInitializer &initializer) {
  auto data = fbb.CreateVector<uint8_t>(
      reinterpret_cast<const uint8_t *>(initializer.data.data()),
      initializer.data.size());
  return dnx::CreateTensorInitializer(fbb, initializer.tensor, data);
}

static flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<denox::dnx::TensorInitializer>>>
serialize_initializers(flatbuffers::FlatBufferBuilder &fbb,
                       memory::span<const TensorInitializer> initializers) {

  memory::vector<flatbuffers::Offset<denox::dnx::TensorInitializer>> offsets(
      initializers.size());
  for (size_t i = 0; i < initializers.size(); ++i) {
    offsets[i] = serialize_initializer(fbb, initializers[i]);
  }

  return fbb.CreateVector<denox::dnx::TensorInitializer>(offsets.data(),
                                                         offsets.size());
}

static flatbuffers::Offset<flatbuffers::Vector<uint32_t>>
serialize_inputs(flatbuffers::FlatBufferBuilder &fbb,
                 memory::span<const uint64_t> inputs) {
  memory::vector<uint32_t> in(inputs.size());
  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = static_cast<uint32_t>(inputs[i]);
  }
  return fbb.CreateVector<uint32_t>(in.data(), in.size());
}

static flatbuffers::Offset<flatbuffers::Vector<uint32_t>>
serialize_outputs(flatbuffers::FlatBufferBuilder &fbb,
                  memory::span<const uint64_t> outputs) {
  memory::vector<uint32_t> out(outputs.size());
  for (size_t o = 0; o < out.size(); ++o) {
    out[o] = static_cast<uint32_t>(outputs[o]);
  }
  return fbb.CreateVector<uint32_t>(out.data(), out.size());
}

static flatbuffers::Offset<denox::dnx::Buffer>
serialize_buffer(flatbuffers::FlatBufferBuilder &fbb, const Buffer &buffer) {
  auto [size_type, size] =
      serialize_scalar(fbb, buffer.size, memory::Dtype::U64);
  return dnx::CreateBuffer(fbb, size_type, size, buffer.alignment);
}

static flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<denox::dnx::Buffer>>>
serialize_buffers(flatbuffers::FlatBufferBuilder &fbb,
                  const memory::span<const Buffer> buffers) {
  memory::vector<flatbuffers::Offset<denox::dnx::Buffer>> offsets(buffers.size());
  for (size_t b = 0; b < buffers.size(); ++b) {
    offsets[b] = serialize_buffer(fbb, buffers[b]);
  }
  return fbb.CreateVector<denox::dnx::Buffer>(offsets.data(), offsets.size());
}

static flatbuffers::Offset<denox::dnx::DescriptorBinding>
serialize_descriptor_binding(flatbuffers::FlatBufferBuilder &fbb,
                             const TensorBinding &binding) {
  auto access = serialize_access(binding.accessFlag);
  return dnx::CreateDescriptorBinding(
      fbb, static_cast<uint16_t>(binding.binding), access,
      static_cast<uint32_t>(binding.tensorId.index));
}

static flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<denox::dnx::DescriptorBinding>>>
serialize_descriptor_bindings(flatbuffers::FlatBufferBuilder &fbb,
                              memory::span<const TensorBinding> bindings) {
  memory::vector<flatbuffers::Offset<denox::dnx::DescriptorBinding>> offsets(
      bindings.size());
  for (size_t b = 0; b < bindings.size(); ++b) {
    offsets[b] = serialize_descriptor_binding(fbb, bindings[b]);
  }
  return fbb.CreateVector<denox::dnx::DescriptorBinding>(offsets.data(),
                                                         offsets.size());
}

static flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<denox::dnx::DescriptorSetBinding>>>
serialize_descriptor_set_bindings(flatbuffers::FlatBufferBuilder &fbb,
                                  memory::span<const TensorBinding> bindings) {
  if (bindings.empty()) {
    return fbb.CreateVector<denox::dnx::DescriptorSetBinding>(nullptr, 0);
  }
  uint32_t maxSet = 0;
  for (const auto &b : bindings) {
    maxSet = std::max(maxSet, b.set);
  }
  uint32_t setCount = maxSet + 1;
  memory::vector<memory::vector<TensorBinding>> setBindings(setCount);
  for (const auto &b : bindings) {
    setBindings[b.set].push_back(b);
  }
  memory::vector<flatbuffers::Offset<denox::dnx::DescriptorSetBinding>> offsets(
      setCount);
  for (uint32_t s = 0; s < setCount; ++s) {
    auto bindings = serialize_descriptor_bindings(fbb, setBindings[s]);
    offsets[s] = dnx::CreateDescriptorSetBinding(fbb, static_cast<uint16_t>(s),
                                                 bindings);
  }
  return fbb.CreateVector<denox::dnx::DescriptorSetBinding>(offsets.data(),
                                                            offsets.size());
}

static flatbuffers::Offset<denox::dnx::PushConstant>
serialize_push_constant(flatbuffers::FlatBufferBuilder &fbb,
                        memory::span<const PushConstant> pushConstants) {
  size_t offset = 0;
  memory::vector<flatbuffers::Offset<denox::dnx::PushConstantField>> offsets(
      pushConstants.size());
  for (size_t p = 0; p < pushConstants.size(); ++p) {
    const auto &pc = pushConstants[p];
    offset = algorithm::align_up(offset, pc.type().alignment());
    auto type = serialize_type(pc.type());

    auto [source_type, source] = serialize_scalar(fbb, pc.sym(), pc.type());
    offsets[p] = dnx::CreatePushConstantField(
        fbb, type, static_cast<uint16_t>(offset), source_type, source);

    offset += pc.type().size();
  }
  assert(offset <= std::numeric_limits<uint16_t>::max());
  uint16_t size = static_cast<uint16_t>(offset);
  auto fields = fbb.CreateVector<denox::dnx::PushConstantField>(offsets.data(),
                                                                offsets.size());
  return dnx::CreatePushConstant(fbb, size, fields);
}

static flatbuffers::Offset<dnx::DispatchInfo>
serialize_dispatch_info(flatbuffers::FlatBufferBuilder &fbb,
                        const ComputeDispatchInfo &info) {

  flatbuffers::Offset<flatbuffers::String> name = 0;
  flatbuffers::Offset<flatbuffers::String> debug_info = 0;
  flatbuffers::Offset<flatbuffers::String> src_path = 0;

  denox::dnx::ScalarSource memory_reads_type = denox::dnx::ScalarSource_NONE;
  flatbuffers::Offset<void> memory_reads = 0;

  denox::dnx::ScalarSource memory_writes_type = denox::dnx::ScalarSource_NONE;
  flatbuffers::Offset<void> memory_writes = 0;

  if (info.name) {
    name = fbb.CreateString(*info.name);
  }
  if (info.debug_info) {
    debug_info = fbb.CreateString(*info.debug_info);
  }
  if (info.srcPath) {
    src_path = fbb.CreateString(info.srcPath->str());
  }
  if (info.memoryReads) {
    auto [type, ptr] =
        serialize_scalar(fbb, *info.memoryReads, memory::Dtype::U64);
    memory_reads_type = type;
    memory_reads = ptr;
  }
  if (info.memoryWrites) {
    auto [type, ptr] =
        serialize_scalar(fbb, *info.memoryWrites, memory::Dtype::U64);
    memory_writes_type = type;
    memory_writes = ptr;
  }

  return dnx::CreateDispatchInfo(fbb, name, debug_info, src_path,
                                 memory_reads_type, memory_reads,
                                 memory_writes_type, memory_writes);
}

static flatbuffers::Offset<denox::dnx::ComputeDispatch>
serialize_dispatch(flatbuffers::FlatBufferBuilder &fbb,
                   const SpvDispatch &dispatch) {
  auto [workgroupCountX_type, workgroupCountX] =
      serialize_scalar(fbb, dispatch.workgroupCountX, memory::Dtype::U32);
  auto [workgroupCountY_type, workgroupCountY] =
      serialize_scalar(fbb, dispatch.workgroupCountY, memory::Dtype::U32);
  auto [workgroupCountZ_type, workgroupCountZ] =
      serialize_scalar(fbb, dispatch.workgroupCountZ, memory::Dtype::U32);
  auto entry_point = fbb.CreateString("main");
  auto bindings = serialize_descriptor_set_bindings(fbb, dispatch.bindings);
  auto pc = serialize_push_constant(fbb, dispatch.pushConstants);
  auto info = serialize_dispatch_info(fbb, dispatch.info);
  return dnx::CreateComputeDispatch(
      fbb, dispatch.binaryId, workgroupCountX_type, workgroupCountX,
      workgroupCountY_type, workgroupCountY, workgroupCountZ_type,
      workgroupCountZ, entry_point, bindings, pc, info);
}

static flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<denox::dnx::ComputeDispatch>>>
serialize_dispatches(flatbuffers::FlatBufferBuilder &fbb,
                     memory::span<const SpvDispatch> dispatches) {

  memory::vector<flatbuffers::Offset<denox::dnx::ComputeDispatch>> offsets(
      dispatches.size());
  for (size_t d = 0; d < dispatches.size(); ++d) {
    offsets[d] = serialize_dispatch(fbb, dispatches[d]);
  }
  return fbb.CreateVector<denox::dnx::ComputeDispatch>(offsets.data(),
                                                       offsets.size());
}

static flatbuffers::Offset<denox::dnx::ShaderBinary>
serialize_shader_binary(flatbuffers::FlatBufferBuilder &fbb,
                        const SpirvBinary &binary) {
  return dnx::CreateShaderBinary(fbb, fbb.CreateVector(binary.spv));
}

static flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<denox::dnx::ShaderBinary>>>
serialize_shader_binaries(flatbuffers::FlatBufferBuilder &fbb,
                          memory::span<const SpirvBinary> binaries) {

  memory::vector<flatbuffers::Offset<denox::dnx::ShaderBinary>> offsets(
      binaries.size());
  for (size_t s = 0; s < binaries.size(); ++s) {
    offsets[s] = serialize_shader_binary(fbb, binaries[s]);
  }
  return fbb.CreateVector<denox::dnx::ShaderBinary>(offsets.data(),
                                                    offsets.size());
}

static denox::dnx::SymIROpCode serialize_symir_opcode(const SymIROpCode code) {
  switch (code) {
  case SymIROpCode::Add_SS:
    return denox::dnx::SymIROpCode_ADD;
  case compiler::SymIROpCode::Add_SC:
    return static_cast<dnx::SymIROpCode>(denox::dnx::SymIROpCode_ADD |
                                         denox::dnx::SymIROpCode_RHSC);
  case compiler::SymIROpCode::Sub_SS:
    return denox::dnx::SymIROpCode_SUB;
  case compiler::SymIROpCode::Sub_SC:
    return static_cast<dnx::SymIROpCode>(denox::dnx::SymIROpCode_SUB |
                                         denox::dnx::SymIROpCode_RHSC);
  case compiler::SymIROpCode::Sub_CS:
    return static_cast<dnx::SymIROpCode>(denox::dnx::SymIROpCode_SUB |
                                         denox::dnx::SymIROpCode_LHSC);
  case compiler::SymIROpCode::Mul_SS:
    return denox::dnx::SymIROpCode_MUL;
  case compiler::SymIROpCode::Mul_SC:
    return static_cast<dnx::SymIROpCode>(denox::dnx::SymIROpCode_MUL |
                                         denox::dnx::SymIROpCode_RHSC);
  case compiler::SymIROpCode::Div_SS:
    return denox::dnx::SymIROpCode_DIV;
  case compiler::SymIROpCode::Div_SC:
    return static_cast<dnx::SymIROpCode>(denox::dnx::SymIROpCode_DIV |
                                         denox::dnx::SymIROpCode_RHSC);
  case compiler::SymIROpCode::Div_CS:
    return static_cast<dnx::SymIROpCode>(denox::dnx::SymIROpCode_DIV |
                                         denox::dnx::SymIROpCode_LHSC);
  case compiler::SymIROpCode::Mod_SS:
    return denox::dnx::SymIROpCode_MOD;
  case compiler::SymIROpCode::Mod_SC:
    return static_cast<dnx::SymIROpCode>(denox::dnx::SymIROpCode_MOD |
                                         denox::dnx::SymIROpCode_RHSC);
  case compiler::SymIROpCode::Mod_CS:
    return static_cast<dnx::SymIROpCode>(denox::dnx::SymIROpCode_MOD |
                                         denox::dnx::SymIROpCode_LHSC);
  case compiler::SymIROpCode::Min_SS:
    return denox::dnx::SymIROpCode_MIN;
  case compiler::SymIROpCode::Min_SC:
    return static_cast<dnx::SymIROpCode>(denox::dnx::SymIROpCode_MIN |
                                         denox::dnx::SymIROpCode_RHSC);
  case compiler::SymIROpCode::Max_SS:
    return denox::dnx::SymIROpCode_MAX;
  case compiler::SymIROpCode::Max_SC:
    return static_cast<dnx::SymIROpCode>(denox::dnx::SymIROpCode_MAX |
                                         denox::dnx::SymIROpCode_RHSC);
  default:
    diag::unreachable();
  }
}

static denox::dnx::SymIROp serialize_symir_op(const SymIROp &op) {
  denox::dnx::SymIROpCode opcode = serialize_symir_opcode(op.opcode);
  return denox::dnx::SymIROp(opcode, op.lhs, op.rhs);
}

static flatbuffers::Offset<flatbuffers::Vector<const denox::dnx::SymIROp *>>
serialize_symir_ops(flatbuffers::FlatBufferBuilder &fbb,
                    const memory::span<const SymIROp> ops) {
  memory::vector<denox::dnx::SymIROp> out(ops.size());
  for (size_t pc = 0; pc < ops.size(); ++pc) {
    out[pc] = serialize_symir_op(ops[pc]);
  }
  return fbb.CreateVectorOfStructs<denox::dnx::SymIROp>(out.data(), out.size());
}

static flatbuffers::Offset<denox::dnx::SymIR>
serialize_symir(flatbuffers::FlatBufferBuilder &fbb, const SymIR &symir) {
  uint16_t varCount = static_cast<uint16_t>(symir.varCount);
  auto ops = serialize_symir_ops(fbb, symir.ops);
  return dnx::CreateSymIR(fbb, varCount, ops);
}

static flatbuffers::Offset<denox::dnx::ValueName>
serialize_value_name(flatbuffers::FlatBufferBuilder &fbb,
                     const NamedValue &namedValue) {
  auto name = fbb.CreateString(namedValue.name);
  auto [value_type, value] =
      serialize_scalar(fbb, namedValue.value, memory::Dtype::I64);
  return dnx::CreateValueName(fbb, name, value_type, value);
}

static flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<denox::dnx::ValueName>>>
serialize_value_names(flatbuffers::FlatBufferBuilder &fbb,
                      memory::span<const NamedValue> namedValues) {
  memory::vector<flatbuffers::Offset<denox::dnx::ValueName>> offsets(
      namedValues.size());
  for (size_t i = 0; i < namedValues.size(); ++i) {
    offsets[i] = serialize_value_name(fbb, namedValues[i]);
  }
  return fbb.CreateVector<denox::dnx::ValueName>(offsets.data(),
                                                 offsets.size());
}

memory::vector<std::byte> serialize(const compiler::SpvSchedule &schedule,
                                    const SymProgram &sprog, const Model &model,
                                    const CompileOptions &options) {

  const unsigned int dnxVersion = options.dnxVersion;

  flatbuffers::FlatBufferBuilder fbb(1 << 16);

  auto version = serialize_version(dnxVersion);
  auto required_features = serialize_required_features(fbb);
  auto model_info = serialize_model_info(fbb, model.meta());
  auto tensors = serialize_tensors(fbb, schedule.tensors);
  auto initializers = serialize_initializers(fbb, schedule.initializers);
  auto inputs = serialize_inputs(fbb, schedule.inputs);
  auto outputs = serialize_outputs(fbb, schedule.outputs);
  auto buffers = serialize_buffers(fbb, schedule.buffers);
  auto dispatches = serialize_dispatches(fbb, schedule.dispatches);
  auto binaries = serialize_shader_binaries(fbb, schedule.binaries);
  auto symir = serialize_symir(fbb, sprog.ir);
  auto value_names = serialize_value_names(fbb, sprog.namedValues);

  auto dnx = dnx::CreateModel(fbb, version, required_features, model_info,
                              tensors, initializers, inputs, outputs, buffers,
                              dispatches, binaries, symir, value_names);
  dnx::FinishModelBuffer(fbb, dnx);
  auto buffer = fbb.Release();

  flatbuffers::Verifier v(buffer.data(), buffer.size());
  if (!dnx::VerifyModelBuffer(v)) {
    DENOX_ERROR("Invalid DNX artefact: failed to verify");
    diag::invalid_state();
  }
  memory::vector<std::byte> outbuf(buffer.size());
  std::memcpy(outbuf.data(), buffer.data(), buffer.size());
  return outbuf;
}

} // namespace denox::compiler
