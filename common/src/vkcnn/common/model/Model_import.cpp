#include "./Model.hpp"
#include "onnx.pb.h"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <filesystem>
#include <fmt/base.h>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace vkcnn {

namespace details {

using opset_version = std::int64_t;
struct OpSetVersions {
  opset_version core_version;

  opset_version operator[](const std::string &domain) {
    assert(map.contains(domain));
    return map[domain];
  }

  std::unordered_map<std::string, opset_version> map;
};

enum class Dtype {
  Undefined,
  Int8,
  Int16,
  Int32,
  Int64,
  Uint8,
  Uint16,
  Uint32,
  Uint64,
  Float64,
  Float32,
  Float16,
  // everything else is not supported (currently)
};

// raw constant payload (weights, biases, const tensors)
struct ConstStorage {
  Dtype type{};
  std::vector<std::byte> raw;
};

class Dim {
  using Rep = std::variant<std::monostate, uint64_t, Sym>;

public:
  static Dim Const(uint64_t v) { return Dim(Rep{v}); }
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

using ShapeVector = std::vector<Dim>;

class UnsignedTensor {
  using Rep = std::variant<uint64_t,              // scalar const
                           Sym,                   // scalar symbolic
                           std::vector<uint64_t>, // vector const
                           std::vector<Sym>       // vector symbolic
                           >;

public:
  explicit UnsignedTensor(uint64_t v) : m_rep(v) {}
  explicit UnsignedTensor(Sym s) : m_rep(std::move(s)) {}
  explicit UnsignedTensor(std::vector<uint64_t> vv) : m_rep(std::move(vv)) {}
  explicit UnsignedTensor(std::vector<Sym> vv) : m_rep(std::move(vv)) {}

  bool isConst() const { return std::holds_alternative<uint64_t>(m_rep); }
  bool isSym() const { return std::holds_alternative<Sym>(m_rep); }
  bool isConstants() const {
    return std::holds_alternative<std::vector<uint64_t>>(m_rep);
  }
  bool isSymbols() const {
    return std::holds_alternative<std::vector<Sym>>(m_rep);
  }

  uint64_t constant() const {
    if (!isConst())
      throw std::logic_error("UIntTensor: not scalar const");
    return std::get<uint64_t>(m_rep);
  }
  const Sym &sym() const {
    if (!isSym())
      throw std::logic_error("UIntTensor: not scalar sym");
    return std::get<Sym>(m_rep);
  }
  const std::vector<uint64_t> &symbols() const {
    if (!isConstants())
      throw std::logic_error("UIntTensor: not vector const");
    return std::get<std::vector<uint64_t>>(m_rep);
  }
  const std::vector<Sym> &constants() const {
    if (!isSymbols())
      throw std::logic_error("UIntTensor: not vector sym");
    return std::get<std::vector<Sym>>(m_rep);
  }

  // convenience
  size_t size() const {
    if (isConstants())
      return constants().size();
    if (isSymbols())
      return symbols().size();
    return 0;
  }

private:
  Rep m_rep;
};

class TensorShape {
  using Rep = std::variant<std::monostate, ShapeVector>;

public:
  static TensorShape Scalar() { return TensorShape(Rep{std::monostate{}}); }
  static TensorShape Vec(ShapeVector s) {
    return TensorShape(Rep{std::move(s)});
  }

  bool isScalar() const {
    return std::holds_alternative<std::monostate>(m_rep);
  }
  bool isVec() const {
    return std::holds_alternative<vkcnn::details::ShapeVector>(m_rep);
  }

  const vkcnn::details::ShapeVector &vec() const {
    if (!isVec())
      throw std::logic_error("TensorShape: not static");
    return std::get<vkcnn::details::ShapeVector>(m_rep);
  }

  // rank helpers
  std::optional<size_t> rankIfShape() const {
    if (isScalar())
      return size_t(0);
    if (isVec())
      return vec().size();
    return std::nullopt; // dynamic: unknown here
  }

private:
  explicit TensorShape(Rep r) : m_rep(std::move(r)) {}

  Rep m_rep;
};

// -----------------------------------------------------------------------------
// Tensor storage kinds: none (runtime), constant payload, or UInt meta-tensor.
class Tensor {
public:
  enum class StorageKind { None, Constant, UInt, Runtime };

private:
  TensorShape m_shape;
  using Storage = std::variant<std::monostate, std::shared_ptr<ConstStorage>,
                               std::shared_ptr<UnsignedTensor>, vkcnn::Tensor>;
  Storage m_store;

public:
  // factories
  static Tensor None(TensorShape s) {
    return Tensor(std::move(s), Storage{std::monostate{}});
  }
  static Tensor Constant(TensorShape s, std::shared_ptr<ConstStorage> c) {
    return Tensor(std::move(s), Storage{std::move(c)});
  }
  static Tensor Unsigned(TensorShape s, std::shared_ptr<UnsignedTensor> u) {
    return Tensor(std::move(s), Storage{std::move(u)});
  }

  static Tensor Runtime(TensorShape s, vkcnn::Tensor tensor) {
    return Tensor(std::move(s), Storage{std::move(tensor)});
  }

  // shape
  const TensorShape &shape() const { return m_shape; }

  // kind checks
  StorageKind kind() const {
    if (std::holds_alternative<std::monostate>(m_store))
      return StorageKind::None;
    if (std::holds_alternative<std::shared_ptr<ConstStorage>>(m_store))
      return StorageKind::Constant;
    if (std::holds_alternative<vkcnn::Tensor>(m_store))
      return StorageKind::Runtime;
    return StorageKind::UInt;
  }
  bool isNone() const { return kind() == StorageKind::None; }
  bool isRuntimeTensor() const { return kind() == StorageKind::Runtime; }
  bool isConstant() const { return kind() == StorageKind::Constant; }
  bool isUnsigned() const { return kind() == StorageKind::UInt; }

  // getters (throw on wrong kind)
  const std::shared_ptr<ConstStorage> &constant() const {
    if (!isConstant())
      throw std::logic_error("Tensor: not a constant tensor");
    return std::get<std::shared_ptr<ConstStorage>>(m_store);
  }
  const std::shared_ptr<UnsignedTensor> &uint() const {
    if (!isUnsigned())
      throw std::logic_error("Tensor: not a UInt meta tensor");
    return std::get<std::shared_ptr<UnsignedTensor>>(m_store);
  }

  const vkcnn::Tensor &runtime() const {
    if (!isRuntimeTensor())
      throw std::logic_error("Tensor: not a UInt meta tensor");
    return std::get<vkcnn::Tensor>(m_store);
  }

private:
  Tensor(TensorShape s, Storage st)
      : m_shape(std::move(s)), m_store(std::move(st)) {}
};

// -----------------------------------------------------------------------------
// One global table (name -> Tensor). Constants are those with kind()==Constant.
// Shape/size/index inputs are UInt meta tensors (kind()==UInt).
struct TensorMap {
  std::unordered_map<std::string, Tensor> map;

  bool has(std::string_view name) const {
    return map.find(std::string(name)) != map.end();
  }
  const Tensor &at(std::string_view name) const {
    auto it = map.find(std::string(name));
    if (it == map.end())
      throw std::out_of_range("TensorMap: missing tensor");
    return it->second;
  }

  Tensor &at(std::string_view name) {
    auto it = map.find(std::string(name));
    if (it == map.end())
      throw std::out_of_range("TensorMap: missing tensor");
    return it->second;
  }
};

using SymMap = std::unordered_map<std::string, Sym>;

struct ImportState {

  std::string model_dir;

  std::int64_t ir_version;
  std::string producer_name;
  std::string producer_version;
  std::string domain;
  std::int64_t model_version;
  OpSetVersions opset_versions;

  TensorMap tensors;

  SymMap symbolMap;

  std::shared_ptr<vkcnn::SymGraph> symGraph;
  vkcnn::Model output;
};

static std::optional<Dtype> parse_data_type(std::int32_t dataType) {
  switch (dataType) {
  case onnx::TensorProto_DataType_UNDEFINED:
    return Dtype::Undefined;
  case onnx::TensorProto_DataType_FLOAT:
    return Dtype::Float32;
  case onnx::TensorProto_DataType_FLOAT16:
    return Dtype::Float16;
  case onnx::TensorProto_DataType_DOUBLE:
    return Dtype::Float64;
  case onnx::TensorProto_DataType_INT8:
    return Dtype::Int8;
  case onnx::TensorProto_DataType_INT16:
    return Dtype::Int16;
  case onnx::TensorProto_DataType_INT32:
    return Dtype::Int32;
  case onnx::TensorProto_DataType_INT64:
    return Dtype::Int64;
  case onnx::TensorProto_DataType_UINT8:
    return Dtype::Uint8;
  case onnx::TensorProto_DataType_UINT16:
    return Dtype::Uint16;
  case onnx::TensorProto_DataType_UINT32:
    return Dtype::Uint32;
  case onnx::TensorProto_DataType_UINT64:
    return Dtype::Uint64;

  case onnx::TensorProto_DataType_BFLOAT16:
  case onnx::TensorProto_DataType_BOOL:
  case onnx::TensorProto_DataType_STRING:
  case onnx::TensorProto_DataType_COMPLEX64:
  case onnx::TensorProto_DataType_COMPLEX128:
  case onnx::TensorProto_DataType_FLOAT8E4M3FN:
  case onnx::TensorProto_DataType_FLOAT8E4M3FNUZ:
  case onnx::TensorProto_DataType_FLOAT8E5M2:
  case onnx::TensorProto_DataType_FLOAT8E5M2FNUZ:
  default:
    return std::nullopt;
  }
}

static std::string_view data_type_to_string(std::int32_t dataType) {
  switch (dataType) {
  case onnx::TensorProto_DataType_UNDEFINED:
    return "undefined";
  case onnx::TensorProto_DataType_FLOAT:
    return "float32";
  case onnx::TensorProto_DataType_FLOAT16:
    return "float16";
  case onnx::TensorProto_DataType_DOUBLE:
    return "float64";
  case onnx::TensorProto_DataType_INT8:
    return "int8";
  case onnx::TensorProto_DataType_INT16:
    return "int16";
  case onnx::TensorProto_DataType_INT32:
    return "int32";
  case onnx::TensorProto_DataType_INT64:
    return "int64";
  case onnx::TensorProto_DataType_UINT8:
    return "uint8";
  case onnx::TensorProto_DataType_UINT16:
    return "uint16";
  case onnx::TensorProto_DataType_UINT32:
    return "uint32";
  case onnx::TensorProto_DataType_UINT64:
    return "uint64";
  case onnx::TensorProto_DataType_BFLOAT16:
    return "bfloat16";
  case onnx::TensorProto_DataType_BOOL:
    return "bool";
  case onnx::TensorProto_DataType_STRING:
    return "string";
  case onnx::TensorProto_DataType_COMPLEX64:
    return "complex64";
  case onnx::TensorProto_DataType_COMPLEX128:
    return "complex128";
  case onnx::TensorProto_DataType_FLOAT8E4M3FN:
    return "float8e4m3fn";
  case onnx::TensorProto_DataType_FLOAT8E4M3FNUZ:
    return "float8e4m3fnuz";
  case onnx::TensorProto_DataType_FLOAT8E5M2:
    return "float8e5m2";
  case onnx::TensorProto_DataType_FLOAT8E5M2FNUZ:
    return "float8e5m2fnuz";
  default:
    throw std::runtime_error("Unexpected data_type");
  }
}

static std::size_t dtype_size(Dtype dtype) {
  switch (dtype) {
  case Dtype::Int8:
  case Dtype::Uint8:
    return 1;
  case Dtype::Int16:
  case Dtype::Uint16:
  case Dtype::Float16:
    return 2;
  case Dtype::Int32:
  case Dtype::Uint32:
  case Dtype::Float32:
    return 4;
  case Dtype::Int64:
  case Dtype::Uint64:
  case Dtype::Float64:
    return 8;
  case Dtype::Undefined:
    throw std::logic_error("Trying to call dtype_size with Dtype::Undefined");
    break;
  }
  throw std::runtime_error("Unexpected dtype");
}

static std::pair<Dtype, std::vector<std::byte>>
get_tensor_data(const ImportState &state, const onnx::TensorProto &tensor) {
  static_assert(std::endian::native == std::endian::little,
                "vkcnn requires a little-endian host.");
  try {
    std::optional<Dtype> dtypeOpt = parse_data_type(tensor.data_type());
    if (!dtypeOpt.has_value()) {
      throw std::runtime_error(
          fmt::format("vkcnn: Unsupported data type: \"{}\"",
                      data_type_to_string(tensor.data_type())));
    }
    Dtype dtype = *dtypeOpt;

    if (dtype == Dtype::Undefined) {
      throw std::runtime_error(fmt::format(
          "vkcnn: Data type of initalizer tensor (\"{}\") cannot be undefined.",
          tensor.name()));
    }

    std::size_t n = 1;
    for (const auto &dim : tensor.dims()) {
      n *= dim; // dim is never negative.
    }
    if (n == 0) {
      return std::make_pair(dtype, std::vector<std::byte>{});
    }
    auto select_data_source = [&]() {
      // Read from external file.
      if (tensor.data_location() == onnx::TensorProto_DataLocation_EXTERNAL) {
        std::string location;
        std::size_t offset = 0;
        std::size_t length = 0;
        for (int i = 0; i < tensor.external_data_size(); ++i) {
          const auto &kv = tensor.external_data(i);
          if (kv.key() == "location") {
            location = kv.value();
          } else if (kv.key() == "offset") {
            offset = static_cast<std::size_t>(std::stoull(kv.value()));
          } else if (kv.key() == "length") {
            length = static_cast<std::size_t>(std::stoull(kv.value()));
          }
        }
        if (location.empty()) {
          throw std::runtime_error("Missing external location");
        }
        auto p = std::filesystem::path(state.model_dir) / location;

        std::ifstream f(p, std::ios::binary);
        if (!f) {
          throw std::runtime_error(
              fmt::format("Failed to open File \"{}\".", p.string()));
        }
        f.seekg(0, std::ios::end);
        const std::size_t fileSize = static_cast<std::size_t>(f.tellg());
        if (offset > fileSize) {
          throw std::runtime_error(fmt::format(
              "External file offset ({}) is past EOF of File \"{}\"", offset,
              p.string()));
        }
        const std::size_t toRead = (length == 0) ? (fileSize - offset) : length;
        std::vector<std::byte> raw(toRead);
        f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        f.read(reinterpret_cast<char *>(raw.data()),
               static_cast<std::streamsize>(toRead));
        if (!f) {
          throw std::runtime_error(fmt::format(
              "Short read. Only read {} bytes of {}", f.gcount(), toRead));
        }

        return raw;
      }

      // Read from tensor.raw_data()
      if (!tensor.raw_data().empty()) {
        std::vector<std::byte> raw;
        raw.assign(
            reinterpret_cast<const std::byte *>(tensor.raw_data().data()),
            reinterpret_cast<const std::byte *>(tensor.raw_data().data() +
                                                tensor.raw_data().size()));
        return raw;
      }

      // Read from int32_data, int64_data or uint64_data (no reinterpret_cast).
      switch (dtype) {
      case Dtype::Int8: {
        const std::size_t size = tensor.int32_data_size();
        if (size != 0) {
          std::vector<std::byte> raw(size * sizeof(std::int8_t));
          std::size_t off = 0;
          for (const auto v : tensor.int32_data()) {
            if (v < std::numeric_limits<std::int8_t>::min() ||
                v > std::numeric_limits<std::int8_t>::max()) {
              throw std::runtime_error(
                  fmt::format("Tensor data out of value range of {}. Value: {}",
                              data_type_to_string(tensor.data_type()), v));
            }
            const std::int8_t t = static_cast<std::int8_t>(v);
            std::memcpy(raw.data() + off, &t, sizeof(t));
            off += sizeof(t);
          }
          return raw;
        }
        break;
      }
      case Dtype::Int16: {
        const std::size_t size = tensor.int32_data_size();
        if (size != 0) {
          std::vector<std::byte> raw(size * sizeof(std::int16_t));
          std::size_t off = 0;
          for (const auto v : tensor.int32_data()) {
            if (v < std::numeric_limits<std::int16_t>::min() ||
                v > std::numeric_limits<std::int16_t>::max()) {
              throw std::runtime_error(
                  fmt::format("Tensor data out of value range of {}. Value: {}",
                              data_type_to_string(tensor.data_type()), v));
            }
            const std::int16_t t = static_cast<std::int16_t>(v);
            std::memcpy(raw.data() + off, &t, sizeof(t));
            off += sizeof(t);
          }
          return raw;
        }
        break;
      }
      case Dtype::Int32: {
        const std::size_t size = tensor.int32_data_size();
        if (size != 0) {
          std::vector<std::byte> raw(size * sizeof(std::int32_t));
          std::memcpy(raw.data(), tensor.int32_data().data(), raw.size());
          return raw;
        }
        break;
      }
      case Dtype::Int64: {
        const std::size_t size = tensor.int64_data_size();
        if (size != 0) {
          std::vector<std::byte> raw(size * sizeof(std::int64_t));
          std::memcpy(raw.data(), tensor.int64_data().data(), raw.size());
          return raw;
        }
        break;
      }
      case Dtype::Uint8: {
        const std::size_t size = tensor.int32_data_size();
        if (size != 0) {
          std::vector<std::byte> raw(size * sizeof(std::uint8_t));
          std::size_t off = 0;
          for (const auto v : tensor.int32_data()) {
            if (v < std::numeric_limits<std::uint8_t>::min() ||
                v > std::numeric_limits<std::uint8_t>::max()) {
              throw std::runtime_error(
                  fmt::format("Tensor data out of value range of {}. Value: {}",
                              data_type_to_string(tensor.data_type()), v));
            }
            const std::uint8_t t = static_cast<std::uint8_t>(v);
            std::memcpy(raw.data() + off, &t, sizeof(t));
            off += sizeof(t);
          }
          return raw;
        }
        break;
      }
      case Dtype::Uint16: {
        const std::size_t size = tensor.int32_data_size();
        if (size != 0) {
          std::vector<std::byte> raw(size * sizeof(std::uint16_t));
          std::size_t off = 0;
          for (const auto v : tensor.int32_data()) {
            if (v < std::numeric_limits<std::uint16_t>::min() ||
                v > std::numeric_limits<std::uint16_t>::max()) {
              throw std::runtime_error(
                  fmt::format("Tensor data out of value range of {}. Value: {}",
                              data_type_to_string(tensor.data_type()), v));
            }
            const std::uint16_t t = static_cast<std::uint16_t>(v);
            std::memcpy(raw.data() + off, &t, sizeof(t));
            off += sizeof(t);
          }
          return raw;
        }
        break;
      }
      case Dtype::Uint32: {
        // Prefer uint64_data (common typed-list carrier for UINT32). Pack down.
        {
          const std::size_t size = tensor.uint64_data_size();
          if (size != 0) {
            std::vector<std::byte> raw(size * sizeof(std::uint32_t));
            std::size_t off = 0;
            for (const auto v : tensor.uint64_data()) {
              if (v > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(fmt::format(
                    "Tensor data out of value range of {}. Value: {}",
                    data_type_to_string(tensor.data_type()), v));
              }
              const std::uint32_t t = static_cast<std::uint32_t>(v);
              std::memcpy(raw.data() + off, &t, sizeof(t));
              off += sizeof(t);
            }
            return raw;
          }
        }
        // Fallback: some exporters might put UINT32 into int32_data.
        {
          const std::size_t size = tensor.int32_data_size();
          if (size != 0) {
            std::vector<std::byte> raw(size * sizeof(std::uint32_t));
            std::size_t off = 0;
            for (const auto v : tensor.int32_data()) {
              if (v < 0) {
                throw std::runtime_error(
                    fmt::format("Negative value in int32_data for UINT32 "
                                "tensor (value: {})",
                                v));
              }
              const std::uint32_t t = static_cast<std::uint32_t>(v);
              std::memcpy(raw.data() + off, &t, sizeof(t));
              off += sizeof(t);
            }
            return raw;
          }
        }
        break;
      }
      case Dtype::Uint64: {
        const std::size_t size = tensor.uint64_data_size();
        if (size != 0) {
          std::vector<std::byte> raw(size * sizeof(std::uint64_t));
          std::memcpy(raw.data(), tensor.uint64_data().data(), raw.size());
          return raw;
        }
        break;
      }
      case Dtype::Float64: {
        const std::size_t size = tensor.double_data_size();
        if (size != 0) {
          std::vector<std::byte> raw(size * sizeof(double));
          std::memcpy(raw.data(), tensor.double_data().data(), raw.size());
          return raw;
        }
        break;
      }
      case Dtype::Float32: {
        const std::size_t size = tensor.float_data_size();
        if (size != 0) {
          std::vector<std::byte> raw(size * sizeof(float));
          std::memcpy(raw.data(), tensor.float_data().data(), raw.size());
          return raw;
        }
        break;
      }
      default:
        break;
      }
      throw std::runtime_error("Failed to find data section.");
    };
    auto raw = select_data_source();
    if (raw.size() % dtype_size(dtype) != 0) {
      throw std::runtime_error(
          fmt::format("Invalid data length. Length ({} bytes) is not divisible "
                      "by the data type size {}",
                      raw.size(), dtype_size(dtype)));
    }
    if ((raw.size() / dtype_size(dtype)) != n) {
      throw std::runtime_error(
          fmt::format("Invalid data length. Expected {} elements ({} bytes), "
                      "but got {} ({} bytes)",
                      n, n * dtype_size(dtype), raw.size() / dtype_size(dtype),
                      raw.size()));
    }
    return std::make_pair(dtype, std::move(raw));
  } catch (const std::runtime_error &e) {
    throw std::runtime_error(fmt::format(
        "Failed to read data of tensor \"{}\" [{}]: {}", tensor.name(),
        data_type_to_string(tensor.data_type()), e.what()));
  }
}

static void import_tensor(ImportState &state, const onnx::TensorProto &tensor) {
  const std::string &name = tensor.name();

  if (tensor.has_segment()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Tensor \"{}\" contains segment, not supported by vkcnn.",
        tensor.name()));
  }

  if (state.tensors.has(name)) {
    fmt::println(
        "vkcnn: [Warning]: Tensor \"{}\" is defined multiple times, ignoring "
        "second occurrence.",
        name);
    return;
  }
  ShapeVector dims;
  dims.reserve(static_cast<size_t>(tensor.dims_size()));
  for (int d = 0; d < tensor.dims_size(); ++d) {
    const int64_t v = tensor.dims(d);
    if (v < 0) {
      throw std::runtime_error(fmt::format(
          "vkcnn: initializer has negative dim for tensor \"{}\"", name));
    }
    dims.push_back(Dim::Const(static_cast<uint64_t>(v)));
  }
  TensorShape shape = TensorShape::Vec(std::move(dims));
  auto [dtype, raw] = get_tensor_data(state, tensor);
  auto store = std::make_shared<ConstStorage>(dtype, std::move(raw));
  auto ctensor = Tensor::Constant(shape, store);
  state.tensors.map.emplace(name, std::move(ctensor));
}

enum ValueInfoImportContext {
  Input,
  Output,
};

static void import_value_info(ImportState &state,
                              const onnx::ValueInfoProto &valueInfo,
                              ValueInfoImportContext context) {
  const std::string &name = valueInfo.name();
  if (name.empty()) {
    throw std::runtime_error("vkcnn: \"\" is not a valid tensor name.");
  }
  if (!valueInfo.has_type()) {
    if (context == ValueInfoImportContext::Input) {
      throw std::runtime_error(fmt::format(
          "vkcnn: input tensor \"{}\" does not define a type.", name));
    } else if (context == ValueInfoImportContext::Output) {
      throw std::runtime_error(fmt::format(
          "vkcnn: output tensor \"{}\" does not define a type.", name));
    } else {
      throw std::logic_error("Unexpected ValueInfoImportContext.");
    }
  }
  const auto &type = valueInfo.type();
  if (type.has_optional_type()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Tensor \"{}\" has unsupported optional_type", name));
  } else if (type.has_sparse_tensor_type()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Tensor \"{}\" has unsupported sparse_tensor_type", name));
  } else if (type.has_map_type()) {
    throw std::runtime_error(
        fmt::format("vkcnn: Tensor \"{}\" has unsupported map_type", name));
  } else if (type.has_sequence_type()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Tensor \"{}\" has unsupported sequence_type", name));
  } else if (type.has_tensor_type()) {
    const auto &tensorType = type.tensor_type();
    // TODO parse tensor type.
    const auto dtypeOpt = parse_data_type(tensorType.elem_type());

    if (!dtypeOpt.has_value()) {
      throw std::runtime_error(
          fmt::format("vkcnn: Tensor \"{}\" has unsuppored tensor elem_type {}",
                      name, data_type_to_string(tensorType.elem_type())));
    }
    const auto dtype = *dtypeOpt;

    if (!tensorType.has_shape()) {
      throw std::runtime_error(
          fmt::format("vkcnn: tensor {} has unknown shape. vkcnn does not "
                      "support dynamic ranks",
                      name));
    }
    const auto &shape = tensorType.shape();
    int rank = shape.dim_size();
    auto parse_shape = [&](const onnx::TensorShapeProto &shape) {
      if (rank == 0) {
        throw std::runtime_error(
            "vkcnn: Input/ Output tensors must not be scalars.");
      }
      std::vector<Dim> dims;
      dims.reserve(rank);
      for (const auto &dim : shape.dim()) {
        if (dim.has_dim_param()) {
          if (context == ValueInfoImportContext::Input) {
            Sym sym = state.symGraph->var(); // <- introduce new symbolic.
            state.symbolMap.emplace(dim.dim_param(), sym);
            dims.push_back(Dim::Symbol(sym));
          } else if (context == ValueInfoImportContext::Output) {
            auto it = state.symbolMap.find(dim.dim_param());
            if (it == state.symbolMap.end()) {
              throw std::runtime_error("vkcnn: Output tensor is dangling (Not "
                                       "produced by any Node).");
            }
            Sym sym = it->second;
            dims.push_back(Dim::Symbol(sym));
          }
        } else {
          auto e = dim.dim_value();
          if (e < 0) {
            throw std::runtime_error(fmt::format(
                "vkcnn: Tensor \"{}\" has negative dimension", name));
          }
          dims.push_back(Dim::Const(e));
        }
      }
      return TensorShape::Vec(dims);
    };
    auto tensorShape = parse_shape(shape);
    assert(tensorShape.isVec());

    auto get_runtime_tensor = [&]() -> vkcnn::Tensor {
      if (context == ValueInfoImportContext::Input) {
        unsigned int c;
        if (tensorShape.vec().size() == 3) {
        }

        // TODO where are the channels, we require the input-channels to be
        // known at compiletime.
        auto constShape = tensorShape.vec();
        // return state.output.input(unsigned int channels);
      } else if (context == ValueInfoImportContext::Output) {
        auto it = state.tensors.map.find(name);
        if (it == state.tensors.map.end()) {
          throw std::runtime_error("output tensor never produced by anything.");
        }
        auto tensor = it->second;
        if (!tensor.isRuntimeTensor()) {
          throw std::runtime_error(
              fmt::format("vkcnn: Output tensor (\"{}\") is not a runtime "
                          "tensor. vkcnn does not support constant outputs.",
                          name));
        }
        auto outputTensorShape = tensor.shape();
        if (outputTensorShape.isScalar()) {
          throw std::runtime_error("vkcnn: Output shape must not be a scalar");
        }
        Dim H_dim;
        Dim W_dim;
        Dim C_dim;
        if (outputTensorShape.vec().size() == 3) {
          C_dim = outputTensorShape.vec()[0];
          H_dim = outputTensorShape.vec()[1];
          W_dim = outputTensorShape.vec()[2];
        } else if (outputTensorShape.vec().size() == 4) {
          C_dim = outputTensorShape.vec()[1];
          H_dim = outputTensorShape.vec()[2];
          W_dim = outputTensorShape.vec()[3];
        } else {
          throw std::runtime_error("vkcnn: Output tensor has invalid shape. "
                                   "Expecting rank 3 or 4, in NCHW layout.");
        }

        // NOTE: the runtime tensors refer to tensor representation of the model
        // that we output from importing.
        unsigned int channels = tensor.runtime().channels();
        Symbolic H = tensor.runtime().height();
        Symbolic W = tensor.runtime().width();
        std::optional<vkcnn::FloatType> datatype = tensor.runtime().type();

        if (C_dim.isSym() &&
            state.symGraph->resolve(C_dim.sym()).isSymbolic()) {
          throw std::runtime_error(
              fmt::format("vkcnn: Output tensor (\"{}\") must have a constant "
                          "channels count.",
                          name));
        } else {
          unsigned C_dim_const;
          if (C_dim.isConst()) {
            C_dim_const = C_dim.value();
          } else {
            C_dim_const = state.symGraph->resolve(C_dim.sym()).constant();
          }
          if (C_dim_const != channels) {
            throw std::runtime_error(
                fmt::format("vkcnn: Output tensor (\"{}\") has invalid channel "
                            "count. Expected {}, Got {}.",
                            name, C_dim_const, channels));
          }
        }

        if (H_dim.isConst() ||
            (H_dim.isSym() &&
             state.symGraph->resolve(H_dim.sym()).isConstant())) {
          if (H.isSymbolic()) {
            throw std::runtime_error(fmt::format(
                "vkcnn: Output tensor (\"{}\") has invalid height. Expected "
                "constant value {}, got symbolic expression.",
                name, H_dim.value()));
          } else { // H is constant.
            assert(H.isConstant());
            auto h = H_dim.isConst()
                         ? H_dim.value()
                         : state.symGraph->resolve(H_dim.sym()).constant();
            if (h != H) {
              throw std::runtime_error(
                  fmt::format("vkcnn: Output tensor (\"{}\") has invalid "
                              "height. Expected {}, Got {}",
                              name, h, H.constant()));
            }
          }
        } else {
          Sym H_dim_sym = state.symGraph->resolve(H_dim.sym());
          if (H_dim_sym != H) {
            if (H.isSymbolic()) {
              throw std::runtime_error(fmt::format(
                  "vkcnn: Output tensor (\"{}\") has invalid height. "
                  "Symbolic expressions do not match.",
                  name));
            } else {
              throw std::runtime_error(fmt::format(
                  "vkcnn: Output tensor (\"{}\") has invalid height. "
                  "Expected symbolic expression, Got constant {}.",
                  name, H.constant()));
            }
          }
        }

        if (W_dim.isConst() ||
            (W_dim.isSym() &&
             state.symGraph->resolve(W_dim.sym()).isConstant())) {
          if (H.isSymbolic()) {
            throw std::runtime_error(fmt::format(
                "vkcnn: Output tensor (\"{}\") has invalid width. Expected "
                "constant value {}, got symbolic expression.",
                name, W_dim.value()));
          } else { // H is constant.
            assert(W.isConstant());
            auto w = W_dim.isConst()
                         ? W_dim.value()
                         : state.symGraph->resolve(W_dim.sym()).constant();
            if (w != W) {
              throw std::runtime_error(
                  fmt::format("vkcnn: Output tensor (\"{}\") has invalid "
                              "width. Expected {}, Got {}",
                              name, w, W.constant()));
            }
          }
        } else {
          Sym W_dim_sym = state.symGraph->resolve(W_dim.sym());
          if (W_dim_sym != W) {
            if (W.isSymbolic()) {
              throw std::runtime_error(
                  fmt::format("vkcnn: Output tensor (\"\") has invalid width. "
                              "Symbolic expressions do not match.",
                              name));
            } else {
              throw std::runtime_error(
                  fmt::format("vkcnn: Output tensor (\"\") has invalid width. "
                              "Expected symbolic expression, Got constant {}.",
                              name, W.constant()));
            }
          }
        }

        if (datatype.has_value()) {
          if (datatype.value() == vkcnn::FloatType::F16 &&
              dtype != Dtype::Float16) {
            throw std::runtime_error("vkcnn: Output tensor type mismatch.");
          } else if (datatype.value() == vkcnn::FloatType::F32 &&
                     dtype != Dtype::Float32) {
            throw std::runtime_error("vkcnn: Output tensor type mismatch.");
          } else if (datatype.value() == vkcnn::FloatType::F64 &&
                     dtype != Dtype::Float64) {
            throw std::runtime_error("vkcnn: Output tensor type mismatch.");
          }
        }
        return tensor.runtime();
        // verify or change datatype if set.
      } 
      throw std::logic_error("Unexpeted ValueInfoImportContext");
    };
    vkcnn::Tensor runtime_tensor = get_runtime_tensor();

    if (context == ValueInfoImportContext::Input) {
      auto tensor =
          Tensor::Runtime(std::move(tensorShape), std::move(runtime_tensor));
      state.tensors.map.emplace(name, std::move(tensor));
    }
  }
}

static void import_graph(ImportState &state, const onnx::GraphProto &graph) {
  if (graph.sparse_initializer_size() != 0) {
    throw std::runtime_error("vkcnn: Model contains sparse initializers are "
                             "not supported by vkcnn.");
  }

  for (const auto &tensor : graph.initializer()) {
    import_tensor(state, tensor);
  }

  std::vector<const onnx::ValueInfoProto *> runtime_inputs;
  for (const auto &in : graph.input()) {
    if (!state.tensors.has(in.name())) {
      runtime_inputs.push_back(&in);
    }
  }
  if (runtime_inputs.size() != 1) {
    throw std::runtime_error(
        fmt::format("vkcnn: Expected exactly one runtime input (image). Got {}",
                    runtime_inputs.size()));
  }

  if (graph.output().size() != 1) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Expected exactly one output. Got {}", graph.output_size()));
  }

  const auto &input = *runtime_inputs.front();
  const auto &output = graph.output(0);

  // We probably need a special function for input / output
  import_value_info(state, input, ValueInfoImportContext::Input);

  // PARSE nodes.

  import_value_info(state, output, ValueInfoImportContext::Output);
}

static void import_model(ImportState &state, const onnx::ModelProto &model) {
  state.ir_version = model.ir_version();
  state.producer_name = model.producer_name();
  state.producer_version = model.producer_version();
  state.domain = model.domain(); // informational
  state.model_version = model.model_version();

  if (model.functions_size() != 0) {
    throw std::runtime_error("vkcnn: ONNX functions are not supported.");
  }
  if (model.opset_import_size() == 0) {
    throw std::runtime_error("vkcnn: missing opset_import.");
  }
  if (!model.has_graph()) {
    throw std::runtime_error("vkcnn: missing top-level graph.");
  }

  state.opset_versions.map.clear();
  for (int i = 0; i < model.opset_import_size(); ++i) {
    const auto &imp = model.opset_import(i);
    const std::string dom = imp.domain().empty() ? "ai.onnx" : imp.domain();
    const opset_version ver = static_cast<opset_version>(imp.version());

    auto it = state.opset_versions.map.find(dom);
    if (it == state.opset_versions.map.end() || it->second < ver) {
      state.opset_versions.map[dom] = ver;
    }
  }

  auto core_it = state.opset_versions.map.find("ai.onnx");
  if (core_it == state.opset_versions.map.end() || core_it->second <= 0) {
    throw std::runtime_error("vkcnn: missing or invalid core opset (ai.onnx).");
  }
  state.opset_versions.core_version = core_it->second;

  for (const auto &kv : state.opset_versions.map) {
    const std::string &dom = kv.first;
    const opset_version ver = kv.second;
    if (dom != "ai.onnx") {
      throw std::runtime_error("vkcnn: unsupported operator set domain \"" +
                               dom + "\" (version " + std::to_string(ver) +
                               ")");
    }
  }

  import_graph(state, model.graph());
}

} // namespace details

Model Model::import(std::string_view path_str) {
  details::ImportState state;

  const std::string path(path_str);
  state.model_dir = std::filesystem::path(path).parent_path();
  state.symGraph = state.output.m_controlBlock->symGraph;

  std::ifstream ifs(path, std::ios::binary);
  if (!ifs)
    throw std::runtime_error("vkcnn: cannot open ONNX file: " + path);
  ::onnx::ModelProto onnx;
  if (!onnx.ParseFromIstream(&ifs)) {
    throw std::runtime_error("vkcnn: Failed to parse ONNX protobuf");
  }

  import_model(state, onnx);
  return state.output;
}

} // namespace vkcnn
