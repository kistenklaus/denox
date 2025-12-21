// #pragma once
//
// #include "denox/common/TensorDataType.hpp"
// #include "denox/common/TensorFormat.hpp"
// #include "denox/common/TensorStorage.hpp"
// #include "denox/symbolic/Sym.hpp"
//
// namespace denox::compiler {
//
// class Model;
// class TensorDescriptor;
//
// class ComputeTensor {
// public:
//   friend Model;
//
//   ComputeTensor(Sym width, Sym height, Sym channels,
//                 TensorDataType dtype = TensorDataType::Auto,
//                 TensorStorage storage = TensorStorage::Optimal,
//                 TensorFormat format = TensorFormat::Optimal)
//       : m_width(width), m_height(height), m_channels(channels),
//         m_storage(storage), m_format(format), m_dtype(dtype) {}
//
//   Sym width() const { return m_width; }
//   Sym height() const { return m_height; }
//   Sym channels() const { return m_channels; }
//
//   TensorStorage storage() const { return m_storage; }
//   void setStorage(TensorStorage storage) { m_storage = storage; }
//   TensorFormat format() const { return m_format; }
//   void setFormat(TensorFormat format) { m_format = format; }
//   TensorDataType type() const { return m_dtype; }
//   void setType(TensorDataType dtype) { m_dtype = dtype; }
//
// private:
//   Sym m_width;
//   Sym m_height;
//   Sym m_channels;
//   TensorStorage m_storage;
//   TensorFormat m_format;
//   TensorDataType m_dtype;
// };
//
// } // namespace denox::compiler
//
// template <> struct fmt::formatter<denox::compiler::ComputeTensor> {
//   // No custom format specifiers
//   constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }
//
//   template <typename FormatContext>
//   auto format(const denox::compiler::ComputeTensor &t,
//               FormatContext &ctx) const {
//     return fmt::format_to(
//         ctx.out(),
//         "{{w={}, h={}, c={}, dtype={}, storage={}, format={}}}",
//         t.width(), t.height(), t.channels(), t.type(), t.storage(), t.format());
//   }
// };
