#pragma once

#include "frontend/onnx/details/values/Map.hpp"
#include "frontend/onnx/details/values/Opaque.hpp"
#include "frontend/onnx/details/values/Optional.hpp"
#include "frontend/onnx/details/values/Sequence.hpp"
#include "frontend/onnx/details/values/SparseTensor.hpp"
#include "frontend/onnx/details/values/Tensor.hpp"
#include "denox/memory/container/variant.hpp"
namespace denox::onnx::details {

enum class ValueKind { Tensor, Optional, Sequence, Map, SparseTensor, Opaque };

class Value {
  using Rep =
      memory::variant<Tensor, Optional, Sequence, Map, SparseTensor, Opaque>;

public:
  static Value FromTensor(Tensor t);
  static Value FromOptional(Optional v = {});
  static Value FromSequence(Sequence v = {});
  static Value FromMap(Map v = {});
  static Value FromSparseTensor(SparseTensor v = {});
  static Value FromOpaque(Opaque v = {});
  ValueKind kind() const;
  bool isTensor() const;
  bool isOptional() const;
  bool isSequence() const;
  bool isMap() const;
  bool isSparseTensor() const;
  bool isOpaque() const;
  const Tensor &tensor() const;
  Tensor &tensor();
  const Optional &optional() const { throwNS("Optional"); }
  const Sequence &sequence() const { throwNS("Sequence"); }
  const Map &map() const { throwNS("Map"); }
  const SparseTensor &sparse() const { throwNS("SparseTensor"); }
  const Opaque &opaque() const { throwNS("Opaque"); }
  static constexpr std::string_view kindName(ValueKind k);

private:
  explicit Value(Rep rep);
  [[noreturn]] static void throwNS(const char *what);

private:
  Rep m_rep;
};

} // namespace denox::onnx::details
