#pragma once

#include "memory/container/vector.hpp"
#include "symbolic/Sym.hpp"
namespace denox::compiler {
class SymGraph;

class SymRemap {
public:
  friend class SymGraph;

  Sym operator[](Sym old) {
    if (old.isSymbolic()) {
      return m_remap[old.sym()];
    } else {
      return old;
    }
  }

private:
  SymRemap(memory::vector<Sym> remap) : m_remap(std::move(remap)) {}
  memory::vector<Sym> m_remap;
};

} // namespace denox::compiler
