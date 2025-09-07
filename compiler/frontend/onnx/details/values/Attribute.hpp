#pragma once

#include "frontend/onnx/details/values/GraphAttr.hpp"
#include "frontend/onnx/details/values/HostTensor.hpp"
#include "memory/container/string_view.hpp"
#include "memory/container/variant.hpp"
#include <cstdint>

namespace onnx {
class AttributeProto;
}

namespace denox::onnx::details {

struct NamedAttribute;

enum class AttributeKind {
  Int,
  Float,
  String,
  Tensor,
  Graph,
  Floats,
  Ints,
  Strings,
  Tensors,
  Graphs,
};

class Attribute {
  using Rep = memory::variant<std::int64_t,          // Int
                              float,                 // Float (ONNX float)
                              memory::string,        // String
                              HostTensor,            // Tensor (host-only)
                              GraphAttr,             // Graph (stub)
                              memory::vector<float>, // Floats
                              memory::vector<std::int64_t>,   // Ints
                              memory::vector<memory::string>, // Strings
                              memory::vector<HostTensor>, // Tensors (host-only)
                              memory::vector<GraphAttr>   // Graphs (stub)
                              >;

public:
  static Attribute Int(std::int64_t v);
  static Attribute Float(float v);
  static Attribute String(std::string v);
  static Attribute Tensor(const HostTensor &t);
  static Attribute Tensor(HostTensor &&t);
  static Attribute Graph(GraphAttr g = {});

  static Attribute Ints(memory::vector<std::int64_t> v);
  static Attribute Floats(memory::vector<float> v);
  static Attribute Strings(memory::vector<memory::string> v);
  static Attribute Tensors(memory::vector<HostTensor> v);
  static Attribute Graphs(memory::vector<GraphAttr> v);
  AttributeKind kind() const;
  bool isInt() const;
  bool isFloat() const;
  bool isString() const;
  bool isTensor() const;
  bool isGraph() const;
  bool isInts() const;
  bool isFloats() const;
  bool isStrings() const;
  bool isTensors() const;
  bool isGraphs() const;
  std::int64_t i() const;
  float f() const;
  const memory::string &s() const;
  const HostTensor &t() const;
  const GraphAttr &g() const;
  const memory::vector<std::int64_t> &ints() const;
  const memory::vector<float> &floats() const;
  const memory::vector<memory::string> &strings() const;
  const memory::vector<HostTensor> &tensors() const;
  const memory::vector<GraphAttr> &graphs() const;
  memory::string_view kindName() const noexcept;

  static NamedAttribute parse(const ::onnx::AttributeProto &attrib,
                              const io::Path &externalDir = {},
                              std::string_view nodeName = "unknown-node");

private:
  explicit Attribute(Rep rep);
  void ensure(AttributeKind k) const;


private:
  Rep m_rep;
};

struct NamedAttribute {
  memory::string name;
  Attribute attribute;
};

memory::string_view AttributeKind_name(AttributeKind k) noexcept;

} // namespace denox::onnx::details
