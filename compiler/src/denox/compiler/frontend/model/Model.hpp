#pragma once

#include "denox/common/AutoPadMode.hpp"
#include "denox/common/FilterMode.hpp"
#include "denox/common/PaddingMode.hpp"
#include "denox/common/PoolFunction.hpp"
#include "denox/common/TensorDataType.hpp"
#include "denox/compiler/frontend/model/ComputeOp.hpp"
#include "denox/compiler/frontend/model/ComputeTensor.hpp"
#include "denox/compiler/frontend/model/ModelControlBlock.hpp"
#include "denox/compiler/frontend/model/Tensor.hpp"
#include "denox/memory/container/string_view.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/memory/tensor/BiasTensor.hpp"
#include "denox/memory/tensor/FilterTensor.hpp"
#include "denox/symbolic/SymGraph.hpp"
#include <memory>

namespace denox::compiler {

class Model {
public:
  Model()
      : m_controlBlock(std::make_unique<details::model::ModelControlBlock>()) {}

  Model(std::unique_ptr<details::model::ModelControlBlock> controlBlock)
      : m_controlBlock(std::move(controlBlock)) {}

  Model(const Model &o)
      : m_controlBlock(std::make_unique<details::model::ModelControlBlock>(
            *o.m_controlBlock)) {}

  TensorHandle input(const std::string &name, Sym width, Sym height, Sym channels,
               TensorDataType type = TensorDataType::Auto);

  TensorHandle conv2d(const TensorHandle &src, //
                memory::FilterTensorConstView W,
                memory::optional<memory::BiasTensorConstView> B,
                AutoPadMode autoPad, memory::uvec2 stride,
                memory::optional<memory::uvec2> padding, memory::uvec2 dilation,
                memory::optional<memory::Dtype> atype = memory::nullopt) const;

  TensorHandle activation(const TensorHandle &src, ActivationFunction func) const;

  TensorHandle upsample(const TensorHandle &src, unsigned int scalingFactor,
                  FilterMode mode) const;

  TensorHandle pool(const TensorHandle &src, memory::uvec2 kernelSize,
              memory::uvec2 padding, memory::uvec2 stride,
              memory::uvec2 dilation, PoolFunction poolFunc) const;

  TensorHandle concat(const TensorHandle &src0, const TensorHandle &src1) const;

  TensorHandle pad(const TensorHandle &src0, Sym left, Sym right, Sym top, Sym bottom,
             PaddingMode mode) const;

  TensorHandle slice(const TensorHandle &src0, Sym left, Sym right, Sym top,
               Sym bottom) const;

  void output(const TensorHandle &src, const std::string &name);

  memory::vector<memory::string> getInputNames() const;

  memory::vector<memory::string> getOutputNames() const;

  memory::optional<TensorHandle> getInput(memory::string_view name) const;

  memory::optional<TensorHandle> getOutput(memory::string_view name) const;

  uint32_t getInputCount() const;
  uint32_t getOutputCount() const;

  std::vector<TensorHandle> getInputs() const;
  std::vector<TensorHandle> getOutputs() const;

  // adds a new name to a already existing value,
  // values might have multiple names!
  void assignValueName(memory::string_view name, Sym value,
                       bool imported = false);

  // create a new value if the there is no matching value name
  Sym requireValueOfName(memory::string_view name, bool imported = false);
  // tries to get a value with a specific name, if not found returns nullopt.
  memory::optional<Sym> getValueByName(memory::string_view valueName);

  const memory::AdjGraph<TensorDescriptor, ComputeOp> &graph() const {
    return m_controlBlock->hypergraph;
  }

  std::span<const NamedValue> valueNames() const;

  const SymGraph &symGraph() const { return m_controlBlock->symGraph; }

  memory::string to_string() const;

private:
  std::unique_ptr<details::model::ModelControlBlock> m_controlBlock;
};

} // namespace denox::compiler

template <>
struct fmt::formatter<denox::compiler::Model>
    : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const denox::compiler::Model &model, FormatContext &ctx) const {
    return fmt::formatter<std::string_view>::format(model.to_string(), ctx);
  }
};
