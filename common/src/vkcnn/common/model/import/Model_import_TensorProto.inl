#pragma once

#include "vkcnn/common/model/import/Model_import_dtype.inl"
#include "vkcnn/common/model/import/Model_import_state.inl"
#include <filesystem>
#include <fstream>
#include <utility>
namespace vkcnn::details {

static HostTensor parse_tensor(const ImportState &state,
                               const onnx::TensorProto &tensor) {
  static_assert(std::endian::native == std::endian::little,
                "vkcnn requires a little-endian host.");

  const std::string &name = tensor.name();

  if (tensor.has_segment()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Tensor \"{}\" contains segment, not supported by vkcnn.",
        name));
  }

  // ---- dtype ----
  const auto dtypeOpt = parse_data_type(tensor.data_type());
  if (!dtypeOpt) {
    throw std::runtime_error(
        fmt::format("vkcnn: Unsupported data type: \"{}\"",
                    data_type_to_string(tensor.data_type())));
  }
  const Dtype dt = *dtypeOpt;
  if (dt == Dtype::Undefined) {
    throw std::runtime_error(fmt::format(
        "vkcnn: initializer tensor (\"{}\") type cannot be undefined.", name));
  }

  // ---- shape -> TensorShape (rank-0 scalar if dims_size()==0) ----
  if (!state.symGraph) {
    throw std::runtime_error("vkcnn: parse_tensor: symGraph is null");
  }
  std::vector<std::uint64_t> dims_u64;
  dims_u64.reserve(static_cast<size_t>(tensor.dims_size()));
  for (int i = 0; i < tensor.dims_size(); ++i) {
    const int64_t d = tensor.dims(i);
    if (d < 0) {
      throw std::runtime_error(fmt::format(
          "vkcnn: initializer has negative dim for tensor \"{}\"", name));
    }
    dims_u64.push_back(static_cast<std::uint64_t>(d));
  }
  TensorShape shape{state.symGraph, std::span<const std::uint64_t>(dims_u64)};

  // element count (rank-0 => 1)
  std::size_t n = 1;
  for (auto v : dims_u64)
    n *= static_cast<std::size_t>(v);

  // ---- helper: read external blob if requested ----
  auto read_external_blob =
      [&](std::size_t &outSize) -> std::vector<std::byte> {
    std::string location;
    std::size_t offset = 0, length = 0;
    for (int i = 0; i < tensor.external_data_size(); ++i) {
      const auto &kv = tensor.external_data(i);
      if (kv.key() == "location")
        location = kv.value();
      else if (kv.key() == "offset")
        offset = static_cast<std::size_t>(std::stoull(kv.value()));
      else if (kv.key() == "length")
        length = static_cast<std::size_t>(std::stoull(kv.value()));
    }
    if (location.empty()) {
      throw std::runtime_error("vkcnn: Missing external location");
    }
    auto p = std::filesystem::path(state.model_dir) / location;
    std::ifstream f(p, std::ios::binary);
    if (!f) {
      throw std::runtime_error(
          fmt::format("vkcnn: Failed to open file \"{}\".", p.string()));
    }
    f.seekg(0, std::ios::end);
    const std::size_t fileSize = static_cast<std::size_t>(f.tellg());
    if (offset > fileSize) {
      throw std::runtime_error(
          fmt::format("vkcnn: External file offset ({}) is past EOF of \"{}\"",
                      offset, p.string()));
    }
    const std::size_t toRead = (length == 0) ? (fileSize - offset) : length;
    std::vector<std::byte> raw(toRead);
    f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    f.read(reinterpret_cast<char *>(raw.data()),
           static_cast<std::streamsize>(toRead));
    if (!f) {
      throw std::runtime_error(fmt::format(
          "vkcnn: Short read. Only read {} bytes of {}", f.gcount(), toRead));
    }
    outSize = toRead;
    return raw;
  };

  // ---- Build HostTensorStorage according to source precedence ----
  std::shared_ptr<HostTensorStorage> storage;

  auto make_from_raw = [&](const void *ptr, std::size_t bytes) {
    if (dt == Dtype::String) {
      // We only support strings via string_data(); raw/external-encoded strings
      // not supported.
      throw std::runtime_error(
          "vkcnn: string initializers must use string_data()");
    }
    const std::size_t elt = dtype_size(dt);
    if (elt == 0) {
      throw std::runtime_error("vkcnn: invalid dtype_size for this dtype");
    }
    if (bytes % elt != 0) {
      throw std::runtime_error(fmt::format(
          "vkcnn: Invalid data length ({} bytes) not divisible by {}", bytes,
          elt));
    }
    const std::size_t count = bytes / elt;
    if (count != n) {
      throw std::runtime_error(
          fmt::format("vkcnn: Invalid data length. Expected {} elements ({} "
                      "bytes), got {} ({} bytes)",
                      n, n * elt, count, bytes));
    }
    storage = std::make_shared<HostTensorStorage>(
        HostTensorStorage::Raw(dt, ptr, bytes));
  };

  // 1) External file?
  if (tensor.data_location() == onnx::TensorProto_DataLocation_EXTERNAL) {
    std::size_t bytes = 0;
    auto raw = read_external_blob(bytes);
    make_from_raw(raw.data(), bytes);
  }
  // 2) raw_data?
  else if (!tensor.raw_data().empty()) {
    const auto &rd = tensor.raw_data();
    make_from_raw(rd.data(), rd.size());
  }
  // 3) typed fields
  else {
    switch (dt) {
    case Dtype::Bool: {
      // ONNX puts bools in int32_data (0/1) commonly.
      const std::size_t sz = static_cast<std::size_t>(tensor.int32_data_size());
      if (n == 0) {
        storage = std::make_shared<HostTensorStorage>(
            HostTensorStorage::Raw(Dtype::Bool, nullptr, 0));
        break;
      }
      if (sz != n) {
        throw std::runtime_error("vkcnn: bool initializer length mismatch");
      }
      std::vector<std::uint8_t> bytes(n);
      for (size_t i = 0; i < n; ++i) {
        bytes[i] =
            tensor.int32_data((int)i) ? std::uint8_t(1) : std::uint8_t(0);
      }
      storage = std::make_shared<HostTensorStorage>(
          HostTensorStorage::Raw(Dtype::Bool, bytes.data(), bytes.size()));
      break;
    }
    case Dtype::Int8: {
      const size_t sz = (size_t)tensor.int32_data_size();
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing int32_data for INT8 initializer");
      std::vector<std::int8_t> v(sz);
      for (size_t i = 0; i < sz; ++i) {
        int32_t x = tensor.int32_data((int)i);
        if (x < std::numeric_limits<int8_t>::min() ||
            x > std::numeric_limits<int8_t>::max())
          throw std::runtime_error("vkcnn: INT8 value out of range");
        v[i] = (int8_t)x;
      }
      if (sz != n)
        throw std::runtime_error("vkcnn: INT8 initializer length mismatch");
      storage = std::make_shared<HostTensorStorage>(HostTensorStorage::Int8(v));
      break;
    }
    case Dtype::Int16: {
      const size_t sz = (size_t)tensor.int32_data_size();
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing int32_data for INT16 initializer");
      std::vector<std::int16_t> v(sz);
      for (size_t i = 0; i < sz; ++i) {
        int32_t x = tensor.int32_data((int)i);
        if (x < std::numeric_limits<int16_t>::min() ||
            x > std::numeric_limits<int16_t>::max())
          throw std::runtime_error("vkcnn: INT16 value out of range");
        v[i] = (int16_t)x;
      }
      if (sz != n)
        throw std::runtime_error("vkcnn: INT16 initializer length mismatch");
      storage =
          std::make_shared<HostTensorStorage>(HostTensorStorage::Int16(v));
      break;
    }
    case Dtype::Int32: {
      const size_t sz = (size_t)tensor.int32_data_size();
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing int32_data for INT32 initializer");
      if (sz != n)
        throw std::runtime_error("vkcnn: INT32 initializer length mismatch");
      storage = std::make_shared<HostTensorStorage>(HostTensorStorage::Int32(
          std::span<const int32_t>(tensor.int32_data().data(), sz)));
      break;
    }
    case Dtype::Int64: {
      const size_t sz = (size_t)tensor.int64_data_size();
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing int64_data for INT64 initializer");
      if (sz != n)
        throw std::runtime_error("vkcnn: INT64 initializer length mismatch");
      storage = std::make_shared<HostTensorStorage>(HostTensorStorage::Int64(
          std::span<const int64_t>(tensor.int64_data().data(), sz)));
      break;
    }
    case Dtype::Uint8: {
      const size_t sz = (size_t)tensor.int32_data_size();
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing int32_data for UINT8 initializer");
      std::vector<std::uint8_t> v(sz);
      for (size_t i = 0; i < sz; ++i) {
        int32_t x = tensor.int32_data((int)i);
        if (x < 0 || x > std::numeric_limits<uint8_t>::max())
          throw std::runtime_error("vkcnn: UINT8 value out of range");
        v[i] = (uint8_t)x;
      }
      if (sz != n)
        throw std::runtime_error("vkcnn: UINT8 initializer length mismatch");
      storage =
          std::make_shared<HostTensorStorage>(HostTensorStorage::Uint8(v));
      break;
    }
    case Dtype::Uint16: {
      const size_t sz = (size_t)tensor.int32_data_size();
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing int32_data for UINT16 initializer");
      std::vector<std::uint16_t> v(sz);
      for (size_t i = 0; i < sz; ++i) {
        int32_t x = tensor.int32_data((int)i);
        if (x < 0 || x > std::numeric_limits<uint16_t>::max())
          throw std::runtime_error("vkcnn: UINT16 value out of range");
        v[i] = (uint16_t)x;
      }
      if (sz != n)
        throw std::runtime_error("vkcnn: UINT16 initializer length mismatch");
      storage =
          std::make_shared<HostTensorStorage>(HostTensorStorage::Uint16(v));
      break;
    }
    case Dtype::Uint32: {
      // Primary: uint64_data carrier
      size_t sz64 = (size_t)tensor.uint64_data_size();
      if (sz64) {
        if (sz64 != n)
          throw std::runtime_error("vkcnn: UINT32 initializer length mismatch");
        std::vector<std::uint32_t> v(sz64);
        for (size_t i = 0; i < sz64; ++i) {
          uint64_t x = tensor.uint64_data((int)i);
          if (x > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("vkcnn: UINT32 value out of range");
          v[i] = (uint32_t)x;
        }
        storage =
            std::make_shared<HostTensorStorage>(HostTensorStorage::Uint32(v));
        break;
      }
      // Fallback: some exporters use int32_data for UINT32
      size_t sz32 = (size_t)tensor.int32_data_size();
      if (sz32 == 0 && n != 0)
        throw std::runtime_error("vkcnn: missing data for UINT32 initializer");
      if (sz32 != n)
        throw std::runtime_error("vkcnn: UINT32 initializer length mismatch");
      std::vector<std::uint32_t> v(sz32);
      for (size_t i = 0; i < sz32; ++i) {
        int32_t x = tensor.int32_data((int)i);
        if (x < 0)
          throw std::runtime_error(
              "vkcnn: negative value in UINT32 initializer");
        v[i] = (uint32_t)x;
      }
      storage =
          std::make_shared<HostTensorStorage>(HostTensorStorage::Uint32(v));
      break;
    }
    case Dtype::Uint64: {
      const size_t sz = (size_t)tensor.uint64_data_size();
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing uint64_data for UINT64 initializer");
      if (sz != n)
        throw std::runtime_error("vkcnn: UINT64 initializer length mismatch");
      storage = std::make_shared<HostTensorStorage>(HostTensorStorage::Uint64(
          std::span<const uint64_t>(tensor.uint64_data().data(), sz)));
      break;
    }
    case Dtype::Float32: {
      const size_t sz = (size_t)tensor.float_data_size();
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing float_data for FLOAT32 initializer");
      if (sz != n)
        throw std::runtime_error("vkcnn: FLOAT32 initializer length mismatch");
      storage = std::make_shared<HostTensorStorage>(HostTensorStorage::F32(
          std::span<const float>(tensor.float_data().data(), sz)));
      break;
    }
    case Dtype::Float64: {
      const size_t sz = (size_t)tensor.double_data_size();
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing double_data for FLOAT64 initializer");
      if (sz != n)
        throw std::runtime_error("vkcnn: FLOAT64 initializer length mismatch");
      storage = std::make_shared<HostTensorStorage>(HostTensorStorage::F64(
          std::span<const double>(tensor.double_data().data(), sz)));
      break;
    }
    case Dtype::Float16: {
      // Usually provided via raw_data; if not present, we consider it missing.
      throw std::runtime_error(
          "vkcnn: FLOAT16 initializer must use raw_data or external_data");
    }
    case Dtype::String: {
      const size_t sz = (size_t)tensor.string_data_size();
      if (n == 0) {
        storage = std::make_shared<HostTensorStorage>(
            HostTensorStorage::TakeOwnership(Dtype::String, nullptr, 0));
        break;
      }
      if (sz != n) {
        throw std::runtime_error("vkcnn: STRING initializer length mismatch");
      }
      // Build vector<string> then storage
      std::vector<std::string> vals(sz);
      for (size_t i = 0; i < sz; ++i) {
        vals[i] = tensor.string_data((int)i);
      }
      storage =
          std::make_shared<HostTensorStorage>(HostTensorStorage::String(vals));
      break;
    }
    default:
      throw std::runtime_error("vkcnn: unsupported initializer dtype");
    }
  }
  // ---- HostTensor (identity view over the shape) ----
  return HostTensor(shape, std::move(storage));
}

static void import_tensor(ImportState &state, const onnx::TensorProto &tensor) {
  const std::string &name = tensor.name();
  if (tensor.has_segment()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Tensor \"{}\" contains segment, not supported by vkcnn.",
        name));
  }
  if (state.tensors.has(name)) {
    fmt::println("vkcnn: [Warning]: Tensor \"{}\" is defined multiple times, "
                 "ignoring second occurrence.",
                 name);
    return;
  }
  HostTensor h = parse_tensor(state, tensor);
  state.tensors.map.emplace(name, Tensor::Host(std::move(h)));
}

} // namespace vkcnn::details
