#include "denox/compiler/frontend/onnx/details/values/HostTensorStorage.hpp"
#include <cstring>
#include <utility>

namespace denox::onnx::details {

HostTensorStorage::HostTensorStorage()
    : m_type(Dtype::Undefined), m_raw(nullptr), m_byteSize(0) {}

HostTensorStorage::HostTensorStorage(HostTensorStorage &&o)
    : m_type(std::exchange(o.m_type, Dtype::Undefined)),
      m_raw(std::exchange(o.m_raw, nullptr)),
      m_byteSize(std::exchange(o.m_byteSize, 0)) {}

HostTensorStorage &HostTensorStorage::operator=(HostTensorStorage &&o) {
  if (this == &o) {
    return *this;
  }
  release();
  std::swap(m_type, o.m_type);
  std::swap(m_raw, o.m_raw);
  std::swap(m_byteSize, o.m_byteSize);
  return *this;
}

void HostTensorStorage::release() {
  if (!m_raw)
    return;
  if (m_type == Dtype::String) {
    const std::size_t n = m_byteSize / sizeof(char *); // not char**
    char **arr = static_cast<char **>(m_raw);
    for (std::size_t i = 0; i < n; ++i)
      free(arr[i]);
  }
  free(m_raw);
  m_raw = nullptr;
  m_byteSize = 0;
  m_type = Dtype::Undefined;
}

HostTensorStorage HostTensorStorage::TakeOwnership(Dtype type, void *raw,
                                                   std::size_t bytes) {
  return HostTensorStorage{type, raw, bytes}; // private ctor already does this
}

HostTensorStorage HostTensorStorage::Raw(Dtype type, const void *raw,
                                         std::size_t byteSize) {
  assert(type != Dtype::String);
  void *ptr = malloc(byteSize);
  std::memcpy(ptr, raw, byteSize);
  return HostTensorStorage{type, ptr, byteSize};
}

HostTensorStorage HostTensorStorage::Bool(memory::span<const bool> values) {
  static_assert(sizeof(std::uint8_t) == 1);
  std::size_t rawSize = values.size() * Dtype::Bool.size();
  void *raw = malloc(rawSize);
  if (Dtype::Bool.size() == sizeof(bool)) {
    std::memcpy(raw, values.data(), values.size_bytes());
  } else {
    for (std::size_t i = 0; i < values.size(); ++i) {
      static_cast<std::uint8_t *>(raw)[i] =
          values[i] ? std::uint8_t(1) : std::uint8_t(0);
    }
  }
  return HostTensorStorage{Dtype::Bool, raw, rawSize};
}

HostTensorStorage
HostTensorStorage::Int8(memory::span<const std::int8_t> values) {
  void *raw = malloc(values.size_bytes());
  std::memcpy(raw, values.data(), values.size_bytes());
  return HostTensorStorage{Dtype::Int8, raw, values.size_bytes()};
}

HostTensorStorage
HostTensorStorage::Int16(memory::span<const std::int16_t> values) {
  void *raw = malloc(values.size_bytes());
  std::memcpy(raw, values.data(), values.size_bytes());
  return HostTensorStorage{Dtype::Int16, raw, values.size_bytes()};
}

HostTensorStorage
HostTensorStorage::Int32(memory::span<const std::int32_t> values) {
  void *raw = malloc(values.size_bytes());
  std::memcpy(raw, values.data(), values.size_bytes());
  return HostTensorStorage{Dtype::Int32, raw, values.size_bytes()};
}

HostTensorStorage
HostTensorStorage::Int64(memory::span<const std::int64_t> values) {
  void *raw = malloc(values.size_bytes());
  std::memcpy(raw, values.data(), values.size_bytes());
  return HostTensorStorage{Dtype::Int64, raw, values.size_bytes()};
}

HostTensorStorage
HostTensorStorage::Uint8(memory::span<const std::uint8_t> values) {
  void *raw = malloc(values.size_bytes());
  std::memcpy(raw, values.data(), values.size_bytes());
  return HostTensorStorage{Dtype::Uint8, raw, values.size_bytes()};
}

HostTensorStorage
HostTensorStorage::Uint16(memory::span<const std::uint16_t> values) {
  void *raw = malloc(values.size_bytes());
  std::memcpy(raw, values.data(), values.size_bytes());
  return HostTensorStorage{Dtype::Uint16, raw, values.size_bytes()};
}

HostTensorStorage
HostTensorStorage::Uint32(memory::span<const std::uint32_t> values) {
  void *raw = malloc(values.size_bytes());
  std::memcpy(raw, values.data(), values.size_bytes());
  return HostTensorStorage{Dtype::Uint32, raw, values.size_bytes()};
}

HostTensorStorage
HostTensorStorage::Uint64(memory::span<const std::uint64_t> values) {
  void *raw = malloc(values.size_bytes());
  std::memcpy(raw, values.data(), values.size_bytes());
  return HostTensorStorage{Dtype::Uint64, raw, values.size_bytes()};
}

HostTensorStorage
HostTensorStorage::Sym(memory::span<const denox::Sym> values) {
  void *raw = malloc(values.size_bytes());
  std::memcpy(raw, values.data(), values.size_bytes());
  return HostTensorStorage{Dtype::Sym, raw, values.size_bytes()};
}

HostTensorStorage
HostTensorStorage::F16(memory::span<const memory::f16> values) {
  void *raw = malloc(values.size_bytes());
  std::memcpy(raw, values.data(), values.size_bytes());
  return HostTensorStorage{Dtype::Float16, raw, values.size_bytes()};
}

HostTensorStorage
HostTensorStorage::F32(memory::span<const memory::f32> values) {
  void *raw = malloc(values.size_bytes());
  std::memcpy(raw, values.data(), values.size_bytes());
  return HostTensorStorage{Dtype::Float32, raw, values.size_bytes()};
}

HostTensorStorage
HostTensorStorage::F64(memory::span<const memory::f64> values) {
  void *raw = malloc(values.size_bytes());
  std::memcpy(raw, values.data(), values.size_bytes());
  return HostTensorStorage{Dtype::Float64, raw, values.size_bytes()};
}

HostTensorStorage
HostTensorStorage::String(memory::span<const memory::string> values) {
  std::size_t rawSize = sizeof(char *) * values.size();
  void *raw = malloc(rawSize);
  for (std::size_t i = 0; i < values.size(); ++i) {
    char *str = static_cast<char *>(malloc(values[i].size() + 1));
    std::memcpy(str, values[i].data(), values[i].size());
    str[values[i].size()] = '\0';
    static_cast<char **>(raw)[i] = str;
  }
  return HostTensorStorage{Dtype::String, raw, rawSize};
}

memory::span<const std::uint8_t> HostTensorStorage::boolean() const {
  assert(m_type == Dtype::Bool);
  return {static_cast<const std::uint8_t *>(m_raw), m_byteSize};
}

memory::span<const std::int8_t> HostTensorStorage::i8() const {
  assert(m_type == Dtype::Int8);
  return {static_cast<const std::int8_t *>(m_raw), m_byteSize};
}

memory::span<const std::int16_t> HostTensorStorage::i16() const {
  assert(m_type == Dtype::Int16);
  return {static_cast<const std::int16_t *>(m_raw),
          m_byteSize / sizeof(std::int16_t)};
}

memory::span<const std::int32_t> HostTensorStorage::i32() const {
  assert(m_type == Dtype::Int32);
  return {static_cast<const std::int32_t *>(m_raw),
          m_byteSize / sizeof(std::int32_t)};
}

memory::span<const std::int64_t> HostTensorStorage::i64() const {
  assert(m_type == Dtype::Int64);
  return {static_cast<const std::int64_t *>(m_raw),
          m_byteSize / sizeof(std::int64_t)};
}

memory::span<const std::uint8_t> HostTensorStorage::u8() const {
  assert(m_type == Dtype::Uint8);
  return {static_cast<const std::uint8_t *>(m_raw), m_byteSize};
}

memory::span<const std::uint16_t> HostTensorStorage::u16() const {
  assert(m_type == Dtype::Uint16);
  return {static_cast<const std::uint16_t *>(m_raw),
          m_byteSize / sizeof(std::uint16_t)};
}

memory::span<const std::uint32_t> HostTensorStorage::u32() const {
  assert(m_type == Dtype::Uint32);
  return {static_cast<const std::uint32_t *>(m_raw),
          m_byteSize / sizeof(std::uint32_t)};
}

memory::span<const std::uint64_t> HostTensorStorage::u64() const {
  assert(m_type == Dtype::Uint64);
  return {static_cast<const std::uint64_t *>(m_raw),
          m_byteSize / sizeof(std::uint64_t)};
}

memory::span<const Sym> HostTensorStorage::sym() const {
  assert(m_type == Dtype::Sym);
  return {static_cast<const denox::Sym *>(m_raw),
          m_byteSize / sizeof(denox::Sym)};
}

memory::span<const memory::f16> HostTensorStorage::f16() const {
  assert(m_type == Dtype::Float16);
  return {static_cast<const memory::f16 *>(m_raw),
          m_byteSize / sizeof(memory::f16)};
}

memory::span<const memory::f32> HostTensorStorage::f32() const {
  assert(m_type == Dtype::Float32);
  return {static_cast<const memory::f32 *>(m_raw),
          m_byteSize / sizeof(memory::f32)};
}

memory::span<const memory::f64> HostTensorStorage::f64() const {
  assert(m_type == Dtype::Float64);
  return {static_cast<const memory::f64 *>(m_raw),
          m_byteSize / sizeof(memory::f64)};
}

memory::span<const char *> HostTensorStorage::strs() const {
  return {static_cast<const char **>(m_raw), m_byteSize / sizeof(char *)};
}

HostTensorStorage::HostTensorStorage(Dtype dtype, void *raw,
                                     std::size_t byteSize)
    : m_type(dtype), m_raw(raw), m_byteSize(byteSize) {}

} // namespace denox::onnx::details
