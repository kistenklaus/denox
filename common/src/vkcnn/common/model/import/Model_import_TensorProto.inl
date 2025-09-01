#pragma once

#include "vkcnn/common/model/import/Model_import_dtype.inl"
#include "vkcnn/common/model/import/Model_import_state.inl"
#include <filesystem>
#include <fstream>
#include <utility>
namespace vkcnn::details {

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

static Tensor parse_tensor(const ImportState &state,
                           const onnx::TensorProto &tensor) {
  const std::string &name = tensor.name();

  if (tensor.has_segment()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Tensor \"{}\" contains segment, not supported by vkcnn.",
        tensor.name()));
  }

  std::optional<ShapeTensor> shape;
  if (tensor.dims_size() != 0) {
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
    shape = ShapeTensor::Tensor(std::move(dims));
  } else {
    shape = ShapeTensor::Scalar();
  }
  auto [dtype, raw] = get_tensor_data(state, tensor);
  RawTensor rawTensor{*shape, dtype, std::move(raw)};
  auto ctensor = Tensor::Raw(std::move(rawTensor));
  return ctensor;
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
  auto ctensor = parse_tensor(state, tensor);
  state.tensors.map.emplace(name, std::move(ctensor));
}

} // namespace vkcnn::details
