#pragma once

#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/memory/tensor/ActivationLayout.hpp"
#include "denox/memory/tensor/BiasTensor.hpp"
#include "denox/memory/tensor/FilterTensor.hpp"
#include "denox/common/AutoPadMode.hpp"
#include "denox/compiler/frontend/model/ComputeOp.hpp"
#include "denox/compiler/frontend/model/ComputeTensor.hpp"
#include "denox/compiler/frontend/model/DynamicInputExtent.hpp"
#include "denox/common/FilterMode.hpp"
#include "denox/compiler/frontend/model/ModelControlBlock.hpp"
#include "denox/common/PaddingMode.hpp"
#include "denox/common/PoolFunction.hpp"
#include "denox/compiler/frontend/model/Tensor.hpp"
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

  Tensor
  input(unsigned int channels, const std::string &name,
        memory::optional<memory::ActivationLayout> layout = memory::nullopt,
        memory::optional<memory::Dtype> type = memory::nullopt,
        memory::optional<Sym> W = memory::nullopt,
        memory::optional<Sym> H = memory::nullopt,
        NamedExtent dynamicExtent = {});

  Tensor conv2d(const Tensor &src, memory::FilterTensorConstView W,
                memory::optional<memory::BiasTensorConstView> B,
                AutoPadMode autoPad, memory::uvec2 stride,
                memory::optional<memory::uvec2> padding, memory::uvec2 dilation,
                memory::optional<memory::Dtype> atype = memory::nullopt) const;

  Tensor activation(const Tensor &src, ActivationFunction func) const;

  Tensor upsample(const Tensor &src, unsigned int scalingFactor,
                  FilterMode mode) const;

  Tensor pool(const Tensor &src, memory::uvec2 kernelSize,
              memory::uvec2 padding, memory::uvec2 stride,
              memory::uvec2 dilation, PoolFunction poolFunc) const;

  Tensor concat(const Tensor &src0, const Tensor &src1) const;

  Tensor pad(const Tensor &src0, Sym left, Sym right, Sym top, Sym bottom,
             PaddingMode mode) const;

  Tensor slice(const Tensor &src0, Sym left, Sym right, Sym top,
               Sym bottom) const;

  void output(const Tensor &src, const std::string &name,
              NamedExtent extentNames = {}) const;

  const memory::AdjGraph<ComputeTensor, ComputeOp> &graph() const {
    return m_controlBlock->hypergraph;
  }

  const SymGraph &symGraph() const { return m_controlBlock->symGraph; }

  Tensor getInput() const;
  Tensor getOutput() const;

  const NamedExtent &getInputExtentNames() const {
    return m_controlBlock->inputExtentNames;
  }

  const NamedExtent &getOutputExtentNames() const {
    return m_controlBlock->outputExtentNames;
  }

  const std::string& getInputName() const {
    return m_controlBlock->inputName;
  }
  const std::string& getOutputName() const  {
    return m_controlBlock->outputName;
  }

  memory::string to_string() const;

private:
  std::unique_ptr<details::model::ModelControlBlock> m_controlBlock;
};

} // namespace denox::compiler
