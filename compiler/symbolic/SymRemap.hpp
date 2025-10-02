#pragma once

#include "memory/container/optional.hpp"
#include "memory/container/vector.hpp"
#include "symbolic/Sym.hpp"
namespace denox::compiler {
class SymGraph;

class SymRemap {
public:
  friend class SymGraph;

  Sym operator[](Sym old) const {
    if (old.isSymbolic()) {
      return *m_remap[old.sym()];
    } else {
      return old;
    }
  }

  Sym operator[](Sym::symbol old) const { return *m_remap[old]; }

private:
  SymRemap(memory::vector<memory::optional<Sym>> remap) : m_remap(std::move(remap)) {}
  memory::vector<memory::optional<Sym>> m_remap;
};

} // namespace denox::compiler
