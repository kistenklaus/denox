#include "frontend/onnx/details/values/Value.hpp"

namespace denox::onnx::details {

Value Value::FromOptional(Optional v) { return Value{Rep{std::move(v)}}; }
Value Value::FromSequence(Sequence v) { return Value{Rep{std::move(v)}}; }
Value Value::FromSparseTensor(SparseTensor v) {
  return Value{Rep{std::move(v)}};
}
Value Value::FromOpaque(Opaque v) { return Value{Rep{std::move(v)}}; }
Value Value::FromTensor(Tensor t) { return Value{Rep{std::move(t)}}; }
Value Value::FromMap(Map v) { return Value{Rep{std::move(v)}}; }
ValueKind Value::kind() const {
  if (std::holds_alternative<Tensor>(m_rep))
    return ValueKind::Tensor;
  if (std::holds_alternative<Optional>(m_rep))
    return ValueKind::Optional;
  if (std::holds_alternative<Sequence>(m_rep))
    return ValueKind::Sequence;
  if (std::holds_alternative<Map>(m_rep))
    return ValueKind::Map;
  if (std::holds_alternative<SparseTensor>(m_rep))
    return ValueKind::SparseTensor;
  if (std::holds_alternative<Opaque>(m_rep))
    return ValueKind::Opaque;
  // unreachable
  assert(false);
  return ValueKind::Opaque;
}
bool Value::isTensor() const { return std::holds_alternative<Tensor>(m_rep); }
bool Value::isOptional() const {
  return std::holds_alternative<Optional>(m_rep);
}
bool Value::isSequence() const {
  return std::holds_alternative<Sequence>(m_rep);
}
bool Value::isMap() const { return std::holds_alternative<Map>(m_rep); }
bool Value::isSparseTensor() const {
  return std::holds_alternative<SparseTensor>(m_rep);
}
bool Value::isOpaque() const { return std::holds_alternative<Opaque>(m_rep); }
const Tensor &Value::tensor() const {
  if (!isTensor())
    throw std::runtime_error("not supported: Value is not a Tensor");
  return std::get<Tensor>(m_rep);
}
Tensor &Value::tensor() {
  if (!isTensor())
    throw std::runtime_error("not supported: Value is not a Tensor");
  return std::get<Tensor>(m_rep);
}
constexpr std::string_view Value::kindName(ValueKind k) {
  switch (k) {
  case ValueKind::Tensor:
    return "Tensor";
  case ValueKind::Optional:
    return "Optional";
  case ValueKind::Sequence:
    return "Sequence";
  case ValueKind::Map:
    return "Map";
  case ValueKind::SparseTensor:
    return "SparseTensor";
  case ValueKind::Opaque:
    return "Opaque";
  }
  return "Unknown";
}
Value::Value(Rep rep) : m_rep(std::move(rep)) {}
void Value::throwNS(const char *what) {
  throw std::runtime_error(std::string("not supported: ") + what);
}
} // namespace denox::onnx::details
