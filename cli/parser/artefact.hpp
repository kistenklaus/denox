#pragma once

#include "parser/artefact/DbArtefact.hpp"
#include "parser/artefact/DnxArtefact.hpp"
#include "parser/artefact/OnnxArtefact.hpp"

#include <cassert>
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
