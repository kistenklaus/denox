#include "dnx/serialize.hpp"
#include "algorithm/align_up.hpp"
#include "compiler/ir/SymTable.hpp"
#include "diag/invalid_state.hpp"
#include "diag/not_implemented.hpp"
#include "diag/unreachable.hpp"
#include "dnx.h"
#include "memory/container/vector.hpp"
#include <fmt/printf.h>

namespace denox::dnx {

flatbuffers::DetachedBuffer serialize(const compiler::CompModel &compModel,
                                      const compiler::SymIR &symIR,
                                      const compiler::SymTable &symTable,
                                      const std::string &inputName,
                                      const std::string &outputName) {
  flatbuffers::FlatBufferBuilder fbb(1 << 16);

  // TODO: add_info.
  // TODO: required_features
  memory::vector<flatbuffers::Offset<Tensor>> tensors;
  tensors.reserve(compModel.tensors.size());
  for (std::size_t t = 0; t < compModel.tensors.size(); ++t) {
    ScalarSource offset_type;
    flatbuffers::Offset<void> offset;

    const auto &tensor = compModel.tensors[t];
    if (tensor.offset.isSymbolic()) {
      offset_type = ScalarSource_symbolic;
      auto symRef =
          CreateSymRef(fbb, static_cast<std::uint32_t>(tensor.offset.sym()));
      offset = symRef.Union();
    } else {

      offset_type = ScalarSource_literal;
      memory::vector<std::uint8_t> literalBytes(sizeof(std::uint64_t));
      std::uint64_t v = static_cast<std::uint64_t>(tensor.offset.constant());

      std::memcpy(literalBytes.data(), &v, sizeof(std::uint64_t));

      auto literalBytesVec = fbb.CreateVector(literalBytes);
      auto literal = CreateScalarLiteral(fbb, ScalarType_U64, literalBytesVec);
      offset = literal.Union();
    }

    ScalarSource size_type;
    flatbuffers::Offset<void> size;
    if (tensor.size.isSymbolic()) {
      size_type = ScalarSource_symbolic;
      auto symRef =
          CreateSymRef(fbb, static_cast<std::uint32_t>(tensor.size.sym()));
      size = symRef.Union();
    } else {
      size_type = ScalarSource_literal;
      std::uint64_t v = static_cast<std::uint64_t>(tensor.size.constant());
      auto literalBytesVec = fbb.CreateVector(
          reinterpret_cast<std::uint8_t *>(&v), sizeof(std::uint64_t));
      auto literal = CreateScalarLiteral(fbb, ScalarType_U64, literalBytesVec);
      size = literal.Union();
    }

    TensorBuilder tensorBuilder(fbb);
    tensorBuilder.add_offset_type(offset_type);
    tensorBuilder.add_offset(offset);
    tensorBuilder.add_buffer(static_cast<std::uint32_t>(tensor.buffer));
    tensorBuilder.add_size_type(size_type);
    tensorBuilder.add_size(size);
    tensors.push_back(tensorBuilder.Finish());
  }
  auto tensorsVec = fbb.CreateVector(tensors);

  memory::vector<flatbuffers::Offset<Buffer>> buffers;
  buffers.reserve(compModel.buffers.size());
  for (std::size_t b = 0; b < compModel.buffers.size(); ++b) {
    const auto &buffer = compModel.buffers[b];
    ScalarSource size_type = ScalarSource_NONE;
    flatbuffers::Offset<void> size;
    if (buffer.size.isSymbolic()) {
      size_type = ScalarSource_symbolic;
      auto symRef =
          CreateSymRef(fbb, static_cast<std::uint32_t>(buffer.size.sym()));
      size = symRef.Union();
    } else {
      size_type = ScalarSource_literal;
      std::uint64_t v = static_cast<std::uint64_t>(buffer.size.constant());
      auto literalBytesVec = fbb.CreateVector(
          reinterpret_cast<const std::uint8_t *>(&v), sizeof(std::uint64_t));
      auto literal = CreateScalarLiteral(fbb, ScalarType_U64, literalBytesVec);
      size = literal.Union();
    }
    BufferBuilder bufferBuilder(fbb);
    bufferBuilder.add_size_type(size_type);
    bufferBuilder.add_size(size);
    bufferBuilder.add_alignment(static_cast<std::uint16_t>(buffer.alignment));
    buffers.push_back(bufferBuilder.Finish());
  }
  auto buffersVec = fbb.CreateVector(buffers);

  memory::vector<flatbuffers::Offset<dnx::ShaderBinary>> binaries;
  binaries.reserve(compModel.shaderBinaries.size());
  for (const auto &shaderBinary : compModel.shaderBinaries) {
    auto spvVec = fbb.CreateVector(shaderBinary.spv);
    binaries.push_back(CreateShaderBinary(fbb, spvVec));
  }

  auto shaderBinariesVec = fbb.CreateVector(binaries);

  memory::vector<flatbuffers::Offset<void>> dispatches;
  memory::vector<std::uint8_t> dispatchTypes;
  dispatches.reserve(compModel.dispatches.size());
  dispatchTypes.reserve(compModel.dispatches.size());
  for (std::size_t d = 0; d < compModel.dispatches.size(); ++d) {
    const auto &dispatch = compModel.dispatches[d];

    auto entryPointStr = fbb.CreateString("main");

    memory::vector<flatbuffers::Offset<DescriptorSetBinding>> setBindings;
    setBindings.reserve(dispatch.setBindings.size());
    for (std::size_t s = 0; s < dispatch.setBindings.size(); ++s) {
      const auto &set = dispatch.setBindings[s];
      memory::vector<flatbuffers::Offset<DescriptorBinding>> bindings;
      bindings.reserve(set.bindings.size());
      for (std::size_t b = 0; b < set.bindings.size(); ++b) {
        const auto &binding = set.bindings[b];
        denox::dnx::Access access;
        switch (binding.access) {
        case compiler::AccessFlag::ReadOnly:
          access = Access_ReadOnly;
          break;
        case compiler::AccessFlag::WriteOnly:
          access = Access_WriteOnly;
          break;
        case compiler::AccessFlag::ReadWrite:
          access = Access_ReadWrite;
          break;
        }
        bindings.push_back(CreateDescriptorBinding(
            fbb, static_cast<std::uint16_t>(binding.binding), access,
            static_cast<std::uint32_t>(binding.tensor)));
      }
      auto bindingsVec = fbb.CreateVector(bindings);
      auto setBinding = CreateDescriptorSetBinding(
          fbb, static_cast<std::uint16_t>(set.set), bindingsVec);
      setBindings.push_back(setBinding);
    }
    auto setBindingsVec = fbb.CreateVector(setBindings);

    std::uint64_t size = 0;
    memory::vector<flatbuffers::Offset<PushConstantField>> pushConstantFields;
    pushConstantFields.reserve(dispatch.pushConstants.size());
    for (std::size_t p = 0; p < dispatch.pushConstants.size(); ++p) {
      const compiler::PushConstant &pc = dispatch.pushConstants[p];
      ScalarType type;
      switch (pc.type().kind()) {
      case memory::DtypeKind::F16:
        type = ScalarType_F16;
        break;
      case memory::DtypeKind::F32:
        type = ScalarType_F32;
        break;
      case memory::DtypeKind::F64:
        type = ScalarType_F64;
        break;
      case memory::DtypeKind::U32:
        type = ScalarType_U32;
        break;
      case memory::DtypeKind::I32:
        type = ScalarType_I32;
        break;
      default:
        compiler::diag::unreachable();
      }
      std::uint64_t offset = algorithm::align_up(size, pc.type().alignment());
      size = offset + pc.type().size();
      ScalarSource source_type;
      flatbuffers::Offset<void> source;
      if (pc.isDynamic()) {
        source_type = ScalarSource_symbolic;
        auto symRef =
            CreateSymRef(fbb, static_cast<std::uint32_t>(pc.dynamic()));
        source = symRef.Union();
      } else {
        source_type = ScalarSource_literal;
        switch (type) {
        case ScalarType_I32: {
          std::int32_t v = pc.i32();
          source =
              CreateScalarLiteral(
                  fbb, type,
                  fbb.CreateVector(reinterpret_cast<const std::uint8_t *>(&v),
                                   sizeof(std::int32_t)))
                  .Union();
          break;
        }
        case ScalarType_U32: {
          std::uint32_t v = pc.u32();
          source =
              CreateScalarLiteral(
                  fbb, type,
                  fbb.CreateVector(reinterpret_cast<const std::uint8_t *>(&v),
                                   sizeof(std::uint32_t)))
                  .Union();
          break;
        }
        case ScalarType_I16:
        case ScalarType_U16:
        case ScalarType_I64:
        case ScalarType_U64:
        case ScalarType_F16:
        case ScalarType_F32:
        case ScalarType_F64:
          compiler::diag::not_implemented();
          break;
        }
      }

      auto field = CreatePushConstantField(
          fbb, type, static_cast<std::uint16_t>(offset), source_type, source);
      pushConstantFields.push_back(field);
    }

    auto fieldsVec = fbb.CreateVector(pushConstantFields);

    PushConstantBuilder pushConstantBuilder(fbb);
    pushConstantBuilder.add_size(static_cast<std::uint16_t>(size));
    pushConstantBuilder.add_fields(fieldsVec);

    auto pushConstant = pushConstantBuilder.Finish();

    std::array<ScalarSource, 3> workgroupCounts_type;
    std::array<flatbuffers::Offset<void>, 3> workgroupCounts;
    for (std::size_t i = 0; i < 3; ++i) {
      compiler::Sym count = dispatch.workgroupCount[i];
      if (count.isSymbolic()) {
        workgroupCounts_type[i] = ScalarSource_symbolic;
        auto symRef =
            CreateSymRef(fbb, static_cast<std::uint32_t>(count.sym()));
        workgroupCounts[i] = symRef.Union();
      } else {
        workgroupCounts_type[i] = ScalarSource_literal;
        std::uint32_t v = static_cast<std::uint32_t>(count.constant());
        auto vbytes = fbb.CreateVector(
            reinterpret_cast<const std::uint8_t *>(&v), sizeof(std::uint32_t));
        auto literal = CreateScalarLiteral(fbb, ScalarType_U32, vbytes);
        workgroupCounts[i] = literal.Union();
      }
    }

    ComputeDispatchBuilder computeDispatchBuilder(fbb);
    computeDispatchBuilder.add_binary_id(dispatch.binaryId);
    computeDispatchBuilder.add_entry_point(entryPointStr);
    computeDispatchBuilder.add_bindings(setBindingsVec);
    computeDispatchBuilder.add_push_constant(pushConstant);
    computeDispatchBuilder.add_workgroup_count_x_type(workgroupCounts_type[0]);
    computeDispatchBuilder.add_workgroup_count_x(workgroupCounts[0]);
    computeDispatchBuilder.add_workgroup_count_y_type(workgroupCounts_type[1]);
    computeDispatchBuilder.add_workgroup_count_y(workgroupCounts[1]);
    computeDispatchBuilder.add_workgroup_count_z_type(workgroupCounts_type[2]);
    computeDispatchBuilder.add_workgroup_count_z(workgroupCounts[2]);

    dispatches.push_back(computeDispatchBuilder.Finish().Union());
    dispatchTypes.push_back(Dispatch_ComputeDispatch);
  }
  auto dispatchesVec = fbb.CreateVector(dispatches);
  auto dispatchTypesVec = fbb.CreateVector(dispatchTypes);

  memory::vector<flatbuffers::Offset<DispatchInfo>> dispatchInfos;
  dispatchInfos.reserve(compModel.dispatches.size());

  constexpr bool ENABLE_DEBUG_INFO = true;
  for (std::size_t d = 0; d < compModel.dispatches.size(); ++d) {
    const auto &dispatch = compModel.dispatches[d];
    if (ENABLE_DEBUG_INFO) {
      const auto &debug_info = dispatch.debug_info;
      const auto &memory_reads = dispatch.memory_reads;
      const auto &memory_writes = dispatch.memory_writes;
      flatbuffers::Offset<flatbuffers::String> debug_info_str;
      flatbuffers::Offset<void> reads;
      ScalarSource reads_type;
      flatbuffers::Offset<void> writes;
      ScalarSource writes_type;
      if (debug_info) {
        debug_info_str = fbb.CreateString(debug_info.value());
      }
      if (memory_reads) {
        if (memory_reads->isSymbolic()) {
          reads = 
              CreateSymRef(fbb, static_cast<uint32_t>(memory_reads->sym()))
                  .Union();
          reads_type = ScalarSource_symbolic;
        } else {
          uint64_t v = static_cast<uint64_t>(memory_reads->constant());
          reads =
              CreateScalarLiteral(
                  fbb, ScalarType_U64,
                  fbb.CreateVector(reinterpret_cast<const uint8_t *>(&v),
                                   sizeof(uint64_t)))
                  .Union();
          reads_type = ScalarSource_literal;
        }
      }
      if (memory_writes) {
        if (memory_writes->isSymbolic()) {
          writes = 
              CreateSymRef(fbb, static_cast<uint32_t>(memory_writes->sym()))
                  .Union();
          writes_type = ScalarSource_symbolic;
        } else {
          uint64_t v = static_cast<uint64_t>(memory_writes->constant());
          writes = 
              CreateScalarLiteral(
                  fbb, ScalarType_U64,
                  fbb.CreateVector(reinterpret_cast<const uint8_t *>(&v),
                                   sizeof(uint64_t)))
                  .Union();
          writes_type = ScalarSource_literal;
        }
      }
      DispatchInfoBuilder info(fbb);
      if (debug_info) {
        info.add_debug_info(debug_info_str);
      }
      if (memory_reads) {
        info.add_memory_reads(reads);
        info.add_memory_reads_type(reads_type);
      }
      if (memory_writes) {
        info.add_memory_writes(writes);
        info.add_memory_writes_type(writes_type);
      }
      dispatchInfos.push_back(info.Finish());
    }
  }

  auto dispatchInfosVec = fbb.CreateVector(dispatchInfos);

  memory::vector<dnx::SymIROp> irops;
  irops.reserve(symIR.ops.size());
  for (std::size_t o = 0; o < symIR.ops.size(); ++o) {
    const auto &op = symIR.ops[o];
    dnx::SymIROpCode opcode;
    switch (op.opcode) {
    case compiler::SymIROpCode::Add_SS:
      opcode = SymIROpCode_ADD;
      break;
    case compiler::SymIROpCode::Add_SC:
      opcode =
          static_cast<dnx::SymIROpCode>(SymIROpCode_ADD | SymIROpCode_RHSC);
      break;
    case compiler::SymIROpCode::Sub_SS:
      opcode = SymIROpCode_SUB;
      break;
    case compiler::SymIROpCode::Sub_SC:
      opcode =
          static_cast<dnx::SymIROpCode>(SymIROpCode_SUB | SymIROpCode_RHSC);
      break;
    case compiler::SymIROpCode::Sub_CS:
      opcode =
          static_cast<dnx::SymIROpCode>(SymIROpCode_SUB | SymIROpCode_LHSC);
      break;
    case compiler::SymIROpCode::Mul_SS:
      opcode = SymIROpCode_MUL;
      break;
    case compiler::SymIROpCode::Mul_SC:
      opcode =
          static_cast<dnx::SymIROpCode>(SymIROpCode_MUL | SymIROpCode_RHSC);
      break;
    case compiler::SymIROpCode::Div_SS:
      opcode = SymIROpCode_DIV;
      break;
    case compiler::SymIROpCode::Div_SC:
      opcode =
          static_cast<dnx::SymIROpCode>(SymIROpCode_DIV | SymIROpCode_RHSC);
      break;
    case compiler::SymIROpCode::Div_CS:
      opcode =
          static_cast<dnx::SymIROpCode>(SymIROpCode_DIV | SymIROpCode_LHSC);
      break;
    case compiler::SymIROpCode::Mod_SS:
      opcode = SymIROpCode_MOD;
      break;
    case compiler::SymIROpCode::Mod_SC:
      opcode =
          static_cast<dnx::SymIROpCode>(SymIROpCode_MOD | SymIROpCode_RHSC);
      break;
    case compiler::SymIROpCode::Mod_CS:
      opcode =
          static_cast<dnx::SymIROpCode>(SymIROpCode_MOD | SymIROpCode_LHSC);
      break;
    case compiler::SymIROpCode::Min_SS:
      opcode = SymIROpCode_MIN;
      break;
    case compiler::SymIROpCode::Min_SC:
      opcode =
          static_cast<dnx::SymIROpCode>(SymIROpCode_MIN | SymIROpCode_RHSC);
      break;
    case compiler::SymIROpCode::Max_SS:
      opcode = SymIROpCode_MAX;
      break;
    case compiler::SymIROpCode::Max_SC:
      opcode =
          static_cast<dnx::SymIROpCode>(SymIROpCode_MAX | SymIROpCode_RHSC);
      break;
    default:
      compiler::diag::unreachable();
    }
    irops.emplace_back(opcode, op.lhs, op.rhs);
  }
  auto iropsVec = fbb.CreateVectorOfStructs(irops.data(), irops.size());

  SymIRBuilder symIRBuilder(fbb);
  symIRBuilder.add_var_count(static_cast<std::uint16_t>(symIR.varCount));
  symIRBuilder.add_ops(iropsVec);
  auto sym_ir = symIRBuilder.Finish();

  memory::vector<flatbuffers::Offset<TensorInitializer>> initializers;

  for (std::size_t i = 0; i < compModel.initializers.size(); ++i) {
    const auto &initializer = compModel.initializers[i];

    auto data = fbb.CreateVector(
        reinterpret_cast<const std::uint8_t *>(initializer.data.data()),
        static_cast<std::size_t>(initializer.data.size()));

    auto tensorInitializer =
        CreateTensorInitializer(fbb, initializer.tensor, data);
    initializers.push_back(tensorInitializer);
  }
  auto initializersVec = fbb.CreateVector(initializers);

  memory::vector<flatbuffers::Offset<TensorInfo>> inputs;
  inputs.reserve(compModel.inputs.size());
  for (std::size_t i = 0; i < compModel.inputs.size(); ++i) {
    const auto &input = compModel.inputs[i];

    ScalarSource channels_type = ScalarSource_literal;
    std::uint32_t c = input.channels;
    flatbuffers::Offset<void> channels =
        CreateScalarLiteral(
            fbb, ScalarType_U32,
            fbb.CreateVector(reinterpret_cast<const std::uint8_t *>(&c),
                             sizeof(std::uint32_t)))
            .Union();

    ScalarSource width_type;
    flatbuffers::Offset<void> width;

    if (input.extent.x.isSymbolic()) {
      width_type = ScalarSource_symbolic;
      auto symRef = CreateSymRef(
          fbb, static_cast<std::uint32_t>(input.extent.x.symbol()));
      width = symRef.Union();
    } else {
      width_type = ScalarSource_literal;
      std::uint32_t v = static_cast<std::uint32_t>(input.extent.x.constant());
      width = CreateScalarLiteral(
                  fbb, ScalarType_U32,
                  fbb.CreateVector(reinterpret_cast<const std::uint8_t *>(&v),
                                   sizeof(std::uint32_t)))
                  .Union();
    }

    ScalarSource height_type;
    flatbuffers::Offset<void> height;
    if (input.extent.y.isSymbolic()) {
      height_type = ScalarSource_symbolic;
      auto symRef = CreateSymRef(
          fbb, static_cast<std::uint32_t>(input.extent.y.symbol()));
      height = symRef.Union();
    } else {
      height_type = ScalarSource_literal;
      std::uint32_t v = static_cast<std::uint32_t>(input.extent.y.constant());
      height = CreateScalarLiteral(
                   fbb, ScalarType_U32,
                   fbb.CreateVector(reinterpret_cast<const std::uint8_t *>(&v),
                                    sizeof(std::uint32_t)))
                   .Union();
    }

    dnx::TensorLayout layout = dnx::TensorLayout_HWC;
    switch (input.layout.kind()) {
    case memory::ActivationLayoutKind::CHW:
      layout = dnx::TensorLayout_CHW;
      break;
    case memory::ActivationLayoutKind::HWC:
      layout = dnx::TensorLayout_HWC;
      break;
    case memory::ActivationLayoutKind::CHWC8:
      layout = dnx::TensorLayout_CHWC8;
      break;
    case memory::ActivationLayoutKind::CHWC4:
    case memory::ActivationLayoutKind::CHWC16:
      compiler::diag::invalid_state();
    default:
      compiler::diag::unreachable();
    }

    auto nameStr = fbb.CreateString(inputName);

    dnx::ScalarType dtype;
    switch (input.dtype.kind()) {
    case memory::DtypeKind::F16:
      dtype = dnx::ScalarType::ScalarType_F16;
      break;
    case memory::DtypeKind::F32:
      dtype = dnx::ScalarType::ScalarType_F32;
      break;
    case memory::DtypeKind::F64:
      dtype = dnx::ScalarType::ScalarType_F64;
      break;
    case memory::DtypeKind::U32:
      dtype = dnx::ScalarType::ScalarType_U32;
      break;
    case memory::DtypeKind::I32:
      dtype = dnx::ScalarType::ScalarType_I32;
      break;
    default:
      compiler::diag::unreachable();
    }

    TensorInfoBuilder builder{fbb};
    builder.add_tensor(static_cast<std::uint32_t>(input.tensor.index));
    builder.add_channels_type(channels_type);
    builder.add_channels(channels);
    builder.add_width_type(width_type);
    builder.add_width(width);
    builder.add_height_type(height_type);
    builder.add_height(height);
    builder.add_layout(layout);
    builder.add_type(dtype);
    builder.add_name(nameStr);

    inputs.push_back(builder.Finish());
  }

  auto inputsVec = fbb.CreateVector(inputs);

  memory::vector<flatbuffers::Offset<TensorInfo>> outputs;
  outputs.reserve(compModel.outputs.size());
  for (std::size_t o = 0; o < compModel.outputs.size(); ++o) {
    const auto &output = compModel.outputs[o];

    ScalarSource channels_type = ScalarSource_literal;
    std::uint32_t c = output.channels;
    flatbuffers::Offset<void> channels =
        CreateScalarLiteral(
            fbb, ScalarType_U32,
            fbb.CreateVector(reinterpret_cast<const std::uint8_t *>(&c),
                             sizeof(std::uint32_t)))
            .Union();

    ScalarSource width_type;
    flatbuffers::Offset<void> width;

    if (output.extent.x.isSymbolic()) {
      width_type = ScalarSource_symbolic;
      auto symRef = CreateSymRef(
          fbb, static_cast<std::uint32_t>(output.extent.x.symbol()));
      width = symRef.Union();
    } else {
      width_type = ScalarSource_literal;
      std::uint32_t v = static_cast<std::uint32_t>(output.extent.x.constant());
      width = CreateScalarLiteral(
                  fbb, ScalarType_U32,
                  fbb.CreateVector(reinterpret_cast<const std::uint8_t *>(&v),
                                   sizeof(std::uint32_t)))
                  .Union();
    }

    ScalarSource height_type;
    flatbuffers::Offset<void> height;
    if (output.extent.y.isSymbolic()) {
      height_type = ScalarSource_symbolic;
      auto symRef = CreateSymRef(
          fbb, static_cast<std::uint32_t>(output.extent.y.symbol()));
      height = symRef.Union();
    } else {
      height_type = ScalarSource_literal;
      std::uint32_t v = static_cast<std::uint32_t>(output.extent.y.constant());
      height = CreateScalarLiteral(
                   fbb, ScalarType_U32,
                   fbb.CreateVector(reinterpret_cast<const std::uint8_t *>(&v),
                                    sizeof(std::uint32_t)))
                   .Union();
    }
    dnx::TensorLayout layout;
    switch (output.layout.kind()) {
    case memory::ActivationLayoutKind::CHW:
      layout = dnx::TensorLayout_CHW;
      break;
    case memory::ActivationLayoutKind::HWC:
      layout = dnx::TensorLayout_HWC;
      break;
    case memory::ActivationLayoutKind::CHWC8:
      layout = dnx::TensorLayout_CHWC8;
      break;
    case memory::ActivationLayoutKind::CHWC4:
    case memory::ActivationLayoutKind::CHWC16:
      compiler::diag::invalid_state();
    }

    auto nameStr = fbb.CreateString(outputName);

    dnx::ScalarType dtype;
    switch (output.dtype.kind()) {
    case memory::DtypeKind::F16:
      dtype = dnx::ScalarType::ScalarType_F16;
      break;
    case memory::DtypeKind::F32:
      dtype = dnx::ScalarType::ScalarType_F32;
      break;
    case memory::DtypeKind::F64:
      dtype = dnx::ScalarType::ScalarType_F64;
      break;
    case memory::DtypeKind::U32:
      dtype = dnx::ScalarType::ScalarType_U32;
      break;
    case memory::DtypeKind::I32:
      dtype = dnx::ScalarType::ScalarType_I32;
      break;
    }

    TensorInfoBuilder builder{fbb};
    builder.add_tensor(static_cast<std::uint32_t>(output.tensor.index));
    builder.add_channels_type(channels_type);
    builder.add_channels(channels);
    builder.add_width_type(width_type);
    builder.add_width(width);
    builder.add_height_type(height_type);
    builder.add_height(height);
    builder.add_layout(layout);
    builder.add_type(dtype);
    builder.add_name(nameStr);

    outputs.push_back(builder.Finish());
  }

  auto outputsVec = fbb.CreateVector(outputs);

  memory::vector<flatbuffers::Offset<denox::dnx::ValueName>> valueNames;
  for (const auto &[sym, name] : symTable.symbolNames) {
    auto nameStr = fbb.CreateString(name);
    flatbuffers::Offset<void> value;
    ScalarSource value_type;
    if (sym.isConstant()) {

      value_type = ScalarSource_literal;
      std::uint64_t v = static_cast<std::uint64_t>(sym.constant());
      value = CreateScalarLiteral(
                  fbb, ScalarType_U64,
                  fbb.CreateVector(reinterpret_cast<const std::uint8_t *>(&v),
                                   sizeof(std::uint64_t)))
                  .Union();
    } else {
      value_type = ScalarSource_symbolic;
      value = CreateSymRef(fbb, static_cast<std::uint32_t>(sym.sym())).Union();
    }

    valueNames.push_back(CreateValueName(fbb, nameStr, value_type, value));
  }

  auto valueNamesVec = fbb.CreateVector(valueNames);

  ModelBuilder mb(fbb);
  mb.add_version(Version::Version_DNX_VERSION_1_0);
  mb.add_tensors(tensorsVec);
  mb.add_buffers(buffersVec);
  mb.add_dispatches_type(dispatchTypesVec);
  mb.add_dispatches(dispatchesVec);
  mb.add_dispatch_infos(dispatchInfosVec);
  mb.add_shader_binaries(shaderBinariesVec);
  mb.add_sym_ir(sym_ir);
  mb.add_initializers(initializersVec);
  mb.add_inputs(inputsVec);
  mb.add_outputs(outputsVec);
  mb.add_value_names(valueNamesVec);

  auto model = mb.Finish();

  FinishModelBuffer(fbb, model);
  flatbuffers::DetachedBuffer detachedBuffer = fbb.Release();
  flatbuffers::Verifier v(detachedBuffer.data(), detachedBuffer.size());
  if (!VerifyModelBuffer(v)) {
    compiler::diag::invalid_state();
  }
  return detachedBuffer;
}

} // namespace denox::dnx
