#pragma once

#include "vkcnn/common/containers/small_vector.hpp"
#include "vkcnn/common/model/Model.hpp"
#include "vkcnn/common/symbolic/Sym.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

#include "vkcnn/common/model/import/Model_import_dtype.inl"

namespace vkcnn::details {

class Dim {
  using Rep = std::variant<std::monostate, uint64_t, Sym>;

public:
  static Dim Const(std::uint64_t v) { return Dim(Rep{v}); }
  static Dim Symbol(Sym s) { return Dim(Rep{std::move(s)}); }

  bool isConst() const { return std::holds_alternative<uint64_t>(m_rep); }
  bool isSym() const { return std::holds_alternative<Sym>(m_rep); }

  uint64_t value() const {
    if (!isConst())
      throw std::logic_error("Dim: not value");
    return std::get<uint64_t>(m_rep);
  }
  const Sym &sym() const {
    if (!isSym())
      throw std::logic_error("Dim: not symbolic");
    return std::get<Sym>(m_rep);
  }

  Dim() : m_rep() {}

private:
  explicit Dim(Rep r) : m_rep(std::move(r)) {}

  Rep m_rep;
};

using ShapeVector = vkcnn::containers::small_vector<Dim, 3>;

class ShapeTensor {
  using Rep = std::variant<std::monostate, ShapeVector>;

public:
  static ShapeTensor Scalar() { return ShapeTensor(Rep{std::monostate{}}); }

  static ShapeTensor Vec(std::size_t s) {
    return ShapeTensor(Rep{ShapeVector{Dim::Const(s)}});
  }

  static ShapeTensor Tensor(ShapeVector s) {
    return ShapeTensor(Rep{std::move(s)});
  }

  bool isScalar() const {
    return std::holds_alternative<std::monostate>(m_rep);
  }
  bool isTensor() const {
    return std::holds_alternative<vkcnn::details::ShapeVector>(m_rep);
  }

  const vkcnn::details::ShapeVector &dims() const {
    if (!isTensor())
      throw std::logic_error("TensorShape: not static");
    return std::get<vkcnn::details::ShapeVector>(m_rep);
  }

  ShapeTensor shape() const {
    if (isScalar()) {
      return ShapeTensor::Scalar();
    } else {
      assert(isTensor());
      return ShapeTensor::Vec(dims().size());
    }
  }

  unsigned int rank() const {
    if (isScalar()) {
      return 0;
    } else {
      assert(isTensor());
      return dims().size();
    }
  }

  ShapeTensor() : m_rep() {}

private:
  explicit ShapeTensor(Rep r) : m_rep(std::move(r)) {}

  Rep m_rep;
};

// raw constant payload (weights, biases, const tensors)
struct RawTensor {
  ShapeTensor shape;
  Dtype type{};
  std::vector<std::byte> raw;
};

struct StringTensor {
  std::string str;

  ShapeTensor shape() const { return ShapeTensor::Scalar(); }
};

class Tensor;

struct ListTensor {
  std::vector<vkcnn::details::Tensor> tensors;

  ShapeTensor shape() const;
};

struct ScalarTensor {
  Dtype dtype;
  union V {
    std::int64_t i;
    std::uint64_t u;
    f32 float32;
    f64 float64;
    f16 float16;
  } v;

  ShapeTensor shape() const { return ShapeTensor::Scalar(); }
};

struct RuntimeTensor {
  unsigned int onnx_rank;
  vkcnn::Tensor tensor;

  ShapeTensor shape() const {
    unsigned int r = onnx_rank == 4 ? 1 : 0;
    ShapeVector dims(onnx_rank);
    if (onnx_rank == 4) {
      dims[0] = Dim::Const(1);
    }
    dims[r + 0] = Dim::Const(tensor.channels());
    auto h = tensor.height();
    if (h.isSymbolic()) {
      dims[r + 1] = Dim::Symbol(*h);
    } else {
      dims[r + 1] = Dim::Const(h.constant());
    }
    auto w = tensor.width();
    if (w.isSymbolic()) {
      dims[r + 2] = Dim::Symbol(*w);
    } else {
      dims[r + 2] = Dim::Const(w.constant());
    }
    return ShapeTensor::Tensor(dims);
  }
};

struct UnknownTensor {};

class Tensor {
public:
  enum class StorageKind { Unknown, Raw, String, Runtime, List, Shape, Scalar };

private:
  using Storage =
      std::variant<UnknownTensor, RawTensor, StringTensor, RuntimeTensor,
                   ListTensor, ShapeTensor, ScalarTensor>;
  std::shared_ptr<Storage> m_store;

public:
  static Tensor Unknown() { return Tensor{Storage{UnknownTensor{}}}; }
  static Tensor Raw(RawTensor raw) { return Tensor{Storage{raw}}; }
  static Tensor Shape(ShapeTensor shape) { return Tensor{Storage{shape}}; }
  static Tensor Scalar(ScalarTensor tensor) { return Tensor(Storage{tensor}); }
  static Tensor Scalar(std::int64_t v) {
    return Tensor(Storage{ScalarTensor{.dtype = Dtype::Int64, .v = {.i = v}}});
  }
  static Tensor Scalar(std::int32_t v) {
    return Tensor(Storage{ScalarTensor{.dtype = Dtype::Int32, .v = {.i = v}}});
  }
  static Tensor Scalar(std::int16_t v) {
    return Tensor(Storage{ScalarTensor{.dtype = Dtype::Int16, .v = {.i = v}}});
  }
  static Tensor Scalar(std::int8_t v) {
    return Tensor(Storage{ScalarTensor{.dtype = Dtype::Int8, .v = {.i = v}}});
  }
  static Tensor Scalar(std::uint8_t v) {
    return Tensor(Storage{ScalarTensor{.dtype = Dtype::Uint8, .v = {.u = v}}});
  }
  static Tensor Scalar(std::uint16_t v) {
    return Tensor(Storage{ScalarTensor{.dtype = Dtype::Uint16, .v = {.u = v}}});
  }
  static Tensor Scalar(std::uint32_t v) {
    return Tensor(Storage{ScalarTensor{.dtype = Dtype::Uint32, .v = {.u = v}}});
  }
  static Tensor Scalar(std::uint64_t v) {
    return Tensor(Storage{ScalarTensor{.dtype = Dtype::Uint64, .v = {.u = v}}});
  }
  static Tensor Scalar(f32 v) {
    return Tensor(
        Storage{ScalarTensor{.dtype = Dtype::Float32, .v = {.float32 = v}}});
  }
  static Tensor Scalar(f16 v) {
    return Tensor(
        Storage{ScalarTensor{.dtype = Dtype::Float16, .v = {.float16 = v}}});
  }
  static Tensor Scalar(f64 v) {
    return Tensor(
        Storage{ScalarTensor{.dtype = Dtype::Float64, .v = {.float64 = v}}});
  }

  static Tensor String(const std::string &str) {
    return Tensor(Storage{StringTensor{str}});
  }

  static Tensor Runtime(vkcnn::Tensor tensor, std::size_t onnx_rank) {
    return Tensor(
        Storage{RuntimeTensor(static_cast<unsigned int>(onnx_rank), tensor)});
  }

  static Tensor List(std::vector<Tensor> list) {
    return Tensor(Storage{ListTensor{std::move(list)}});
  }

  // shape
  ShapeTensor shape() const {
    switch (kind()) {
    case StorageKind::Unknown:
      throw std::logic_error("Accessing shape of unknown tensor");
    case StorageKind::Raw:
      return raw().shape;
    case StorageKind::String:
      return string().shape();
    case StorageKind::Runtime:
      return runtime().shape();
    case StorageKind::List:
      return list().shape();
    case StorageKind::Shape:
      return shapeTensor().shape();
    case StorageKind::Scalar:
      return scalar().shape();
      break;
    }
    throw std::logic_error("unreachable");
  }

  // kind checks
  StorageKind kind() const {
    if (std::holds_alternative<UnknownTensor>(*m_store))
      return StorageKind::Unknown;
    if (std::holds_alternative<RawTensor>(*m_store))
      return StorageKind::Raw;
    if (std::holds_alternative<RuntimeTensor>(*m_store))
      return StorageKind::Runtime;
    if (std::holds_alternative<StringTensor>(*m_store))
      return StorageKind::String;
    if (std::holds_alternative<ListTensor>(*m_store))
      return StorageKind::List;
    if (std::holds_alternative<ShapeTensor>(*m_store))
      return StorageKind::Shape;
    if (std::holds_alternative<ScalarTensor>(*m_store))
      return StorageKind::Scalar;
    throw std::logic_error("unreachable");
  }
  bool isRuntimeTensor() const { return kind() == StorageKind::Runtime; }
  bool isRaw() const { return kind() == StorageKind::Raw; }
  bool isString() const { return kind() == StorageKind::String; }
  bool isShape() const { return kind() == StorageKind::Shape; }
  bool isList() const { return kind() == StorageKind::List; }
  bool isScalar() const { return kind() == StorageKind::Scalar; }
  bool isUnknown() const { return kind() == StorageKind::Unknown; }

  // getters (throw on wrong kind)
  const RawTensor &raw() const {
    if (!isRaw())
      throw std::logic_error("Tensor: not a constant tensor");
    return std::get<RawTensor>(*m_store);
  }

  const RuntimeTensor &runtime() const {
    if (!isRuntimeTensor())
      throw std::logic_error("Tensor: not a runtime tensor");
    return std::get<RuntimeTensor>(*m_store);
  }

  RuntimeTensor &runtime() {
    if (!isRuntimeTensor())
      throw std::logic_error("Tensor: not a runtime tensor");
    return std::get<RuntimeTensor>(*m_store);
  }

  const StringTensor &string() const {
    if (!isString())
      throw std::logic_error("Tensor: not a string tensor");
    return std::get<StringTensor>(*m_store);
  }

  const ListTensor &list() const {
    if (!isList())
      throw std::logic_error("Tensor: not a list tensor");
    return std::get<ListTensor>(*m_store);
  }

  const ShapeTensor &shapeTensor() const {
    if (!isShape())
      throw std::logic_error("Tensor: not a list tensor");
    return std::get<ShapeTensor>(*m_store);
  }

  const ScalarTensor &scalar() const {
    if (!isScalar())
      throw std::logic_error("Tensor: not a list tensor");
    return std::get<ScalarTensor>(*m_store);
  }

private:
  Tensor(Storage st) : m_store(std::make_shared<Storage>(std::move(st))) {}
};

ShapeTensor ListTensor::shape() const {
  return ShapeTensor::Vec(tensors.size());
}

} // namespace vkcnn::details
