#pragma once

#include "parser/artefact/DbArtefact.hpp"
#include "parser/artefact/DnxArtefact.hpp"
#include "parser/artefact/OnnxArtefact.hpp"

#include <cassert>
#include <fmt/core.h>
#include <variant>

enum class ArtefactKind { Onnx, Dnx, Database };

class Artefact {
public:
  Artefact(OnnxArtefact onnx) noexcept : m_value(std::move(onnx)) {}

  Artefact(DnxArtefact dnx) noexcept : m_value(std::move(dnx)) {}

  Artefact(DbArtefact db) noexcept : m_value(std::move(db)) {}

  ArtefactKind kind() const noexcept {
    if (std::holds_alternative<OnnxArtefact>(m_value))
      return ArtefactKind::Onnx;
    if (std::holds_alternative<DnxArtefact>(m_value))
      return ArtefactKind::Dnx;
    if (std::holds_alternative<DbArtefact>(m_value))
      return ArtefactKind::Database;

    std::abort();
  }

  const OnnxArtefact &onnx() const noexcept {
    assert(kind() == ArtefactKind::Onnx);
    return std::get<OnnxArtefact>(m_value);
  }

  const DnxArtefact &dnx() const noexcept {
    assert(kind() == ArtefactKind::Dnx);
    return std::get<DnxArtefact>(m_value);
  }

  const DbArtefact &database() const noexcept {
    assert(kind() == ArtefactKind::Database);
    return std::get<DbArtefact>(m_value);
  }

private:
  std::variant<OnnxArtefact, DnxArtefact, DbArtefact> m_value;
};

template <>
struct fmt::formatter<ArtefactKind> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(ArtefactKind kind, FormatContext& ctx) const {
    std::string_view name;

    switch (kind) {
      case ArtefactKind::Onnx:
        name = "onnx model";
        break;
      case ArtefactKind::Dnx:
        name = "dnx model";
        break;
      case ArtefactKind::Database:
        name = "database";
        break;
    }

    return fmt::formatter<std::string_view>::format(name, ctx);
  }
};

template <>
struct fmt::formatter<Artefact> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const Artefact& artefact, FormatContext& ctx) const {
    switch (artefact.kind()) {
      case ArtefactKind::Onnx:
        return fmt::format_to(ctx.out(), "onnx model");
      case ArtefactKind::Dnx:
        return fmt::format_to(ctx.out(), "dnx model");
      case ArtefactKind::Database:
        return fmt::format_to(
          ctx.out(), "database ({})",
          artefact.database().endpoint);
    }
    std::abort();
  }
};
