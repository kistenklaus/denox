#include "denox/compiler/implement/OpImpl.hpp"
#include "denox/common/TensorDataType.hpp"
#include "denox/compiler/implement/ComputeDispatchBuilder.hpp"
#include "denox/compiler/implement/SuperGraphBuilder.hpp"
#include <stdexcept>

namespace denox::compiler {

TensorId OpImpl::createParameter(const memory::FilterDescriptor &descriptor,
                                 memory::FilterTensorConstView data,
                                 TensorStorage storage, TensorFormat format) {
  size_t bytes = descriptor.byteSize();
  uint16_t alignment;
  if (descriptor.layout.isVectorized()) {
    alignment = 16;
  } else {
    // TODO should probably be a call to align_of
    alignment = static_cast<uint16_t>(descriptor.type.alignment());
  }
  TensorId id = m_superBuilder->createTensor(Sym::Const(bytes), alignment);

  m_superBuilder->m_tensors[id.index].info.storage = storage;
  m_superBuilder->m_tensors[id.index].info.format = format;
  m_superBuilder->m_tensors[id.index].info.type =
      tensor_data_type_from_memory_type(descriptor.type);

  if (m_superBuilder->m_writeParameters) {
    auto bytes = m_superBuilder->m_paramCache.convert(descriptor, data);
    m_parameters.push_back(Parameter{
        .tensorId = id,
        .lazyValue = [bytes]() { return *bytes; },
    });
  }
  return id;
}

TensorId OpImpl::createParameter(const memory::BiasDescriptor &descriptor,
                                 memory::BiasTensorConstView data,
                                 TensorStorage storage, TensorFormat format) {
  size_t bytes = descriptor.byteSize();
  uint16_t alignment;
  if (descriptor.layout.isVectorized()) {
    alignment = 16;
  } else {
    alignment = static_cast<uint16_t>(descriptor.type.alignment());
  }
  TensorId id = m_superBuilder->createTensor(Sym::Const(bytes), alignment);

  m_superBuilder->m_tensors[id.index].info.storage = storage;
  m_superBuilder->m_tensors[id.index].info.format = format;
  m_superBuilder->m_tensors[id.index].info.type =
      tensor_data_type_from_memory_type(descriptor.type);

  if (m_superBuilder->m_writeParameters) {
    auto bytes = m_superBuilder->m_paramCache.convert(descriptor, data);
    m_parameters.push_back(Parameter{
        .tensorId = id,
        .lazyValue = [bytes]() { return *bytes; },
    });
  }
  return id;
}

TensorId OpImpl::createParameter(size_t elemCount, TensorDataType dtype,
                                 TensorStorage storage, TensorFormat format,
                                 std::function<std::vector<std::byte>()> value,
                                 uint16_t alignment) {
  size_t bytes = elemCount * size_of(dtype);
  TensorId id = m_superBuilder->createTensor(Sym::Const(bytes), alignment);
  m_superBuilder->m_tensors[id.index].info.storage = storage;
  m_superBuilder->m_tensors[id.index].info.format = format;
  m_superBuilder->m_tensors[id.index].info.type = dtype;

  if (m_superBuilder->m_writeParameters) {
    m_parameters.push_back(Parameter{
        .tensorId = id,
        .lazyValue = value,
    });
  }
}

ComputeDispatchBuilder
OpImpl::registerDispatch(spirv::GlslCompilerInstance glsl, Sym wgX, Sym wgY,
                         Sym wgZ) {
  size_t index = m_dispatches.size();
  m_dispatches.push_back(ComputeDispatch{
      .glsl = std::move(glsl),
      .pushConstants = {},
      .workgroupCountX = wgX,
      .workgroupCountY = wgY,
      .workgroupCountZ = wgZ,
      .bindings = {},
      .info = {},
  });
  return ComputeDispatchBuilder(index, this);
}

void OpImpl::createImplicitConcatConstrain(memory::NodeId src0,
                                           memory::NodeId src1,
                                           memory::NodeId dst) {
  memory::optional<TensorId> tensorSrc0;
  if (*src0 < m_superBuilder->m_nodeTensorMapping.size()) {
    tensorSrc0 = m_superBuilder->m_nodeTensorMapping[*src0];
  }
  memory::optional<TensorId> tensorSrc1;
  if (*src1 < m_superBuilder->m_nodeTensorMapping.size()) {
    tensorSrc1 = m_superBuilder->m_nodeTensorMapping[*src1];
  }
  memory::optional<TensorId> tensorDst;
  if (*dst < m_superBuilder->m_nodeTensorMapping.size()) {
    tensorDst = m_superBuilder->m_nodeTensorMapping[*dst];
  }
  if (tensorSrc0.has_value() && tensorSrc1.has_value() &&
      tensorDst.has_value()) {
    m_memoryConstrains.emplace_back(*tensorSrc0, *tensorSrc1, *tensorDst);
  } else {
    throw std::runtime_error(
        "Failed to create memory constrain, invalid node ids.");
  }
}

void OpImpl::finish() {
  m_superBuilder->m_graph.addEdge(
      m_inputs, m_output,
      SuperGraphEdge{
          .dispatches = std::move(m_dispatches),
          .memoryConstrains = std::move(m_memoryConstrains),
          .parameters = std::move(m_parameters),
      });
  // m_superBuilder = nullptr;
  // m_inputs.clear();
  // m_output = memory::NodeId{}; // <- null
}

} // namespace denox::compiler
