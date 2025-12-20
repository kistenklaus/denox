#pragma once

#include "denox/memory/container/span.hpp"
#include "denox/memory/container/string.hpp"
#include "denox/memory/dtype/f16.hpp"
#include "denox/memory/dtype/f32.hpp"
#include "denox/memory/dtype/f64.hpp"
#include "denox/symbolic/Sym.hpp"

#include "denox/compiler/frontend/onnx/details/values/Dtype.hpp"

namespace denox::onnx::details {

class HostTensorStorage {
public:
  HostTensorStorage();
  HostTensorStorage(const HostTensorStorage &) = delete;
  HostTensorStorage &operator=(const HostTensorStorage &) = delete;
  HostTensorStorage(HostTensorStorage &&o);
  HostTensorStorage &operator=(HostTensorStorage &&o);
  ~HostTensorStorage() { release(); }
  void release();

  static HostTensorStorage TakeOwnership(Dtype type, void *raw,
                                         std::size_t bytes);
  static HostTensorStorage Raw(Dtype type, const void *raw,
                               std::size_t byteSize);
  static HostTensorStorage Bool(memory::span<const bool> values);
  static HostTensorStorage Int8(memory::span<const std::int8_t> values);
  static HostTensorStorage Int16(memory::span<const std::int16_t> values);
  static HostTensorStorage Int32(memory::span<const std::int32_t> values);
  static HostTensorStorage Int64(memory::span<const std::int64_t> values);
  static HostTensorStorage Uint8(memory::span<const std::uint8_t> values);
  static HostTensorStorage Uint16(memory::span<const std::uint16_t> values);
  static HostTensorStorage Uint32(memory::span<const std::uint32_t> values);
  static HostTensorStorage Uint64(memory::span<const std::uint64_t> values);
  static HostTensorStorage Sym(memory::span<const Sym> values);
  static HostTensorStorage F16(memory::span<const memory::f16> values);
  static HostTensorStorage F32(memory::span<const memory::f32> values);
  static HostTensorStorage F64(memory::span<const memory::f64> values);
  static HostTensorStorage String(memory::span<const memory::string> values);

  memory::span<const std::uint8_t> boolean() const;
  memory::span<const std::int8_t> i8() const;
  memory::span<const std::int16_t> i16() const;
  memory::span<const std::int32_t> i32() const;
  memory::span<const std::int64_t> i64() const;
  memory::span<const std::uint8_t> u8() const;
  memory::span<const std::uint16_t> u16() const;
  memory::span<const std::uint32_t> u32() const;
  memory::span<const std::uint64_t> u64() const;
  memory::span<const denox::Sym> sym() const;
  memory::span<const memory::f16> f16() const;
  memory::span<const memory::f32> f32() const;
  memory::span<const memory::f64> f64() const;
  memory::span<const char *> strs() const;

  const void *data() const { return m_raw; }
  void *data() { return m_raw; }
  Dtype type() const { return m_type; }

private:
  explicit HostTensorStorage(Dtype dtype, void *raw, std::size_t byteSize);

private:
  Dtype m_type;
  void *m_raw;
  std::size_t m_byteSize;
};

} // namespace denox::onnx::details
