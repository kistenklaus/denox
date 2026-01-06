#include "denox/compiler/assumed_symeval/assumed_symeval.hpp"
#include "denox/memory/container/dynamic_bitset.hpp"
#include "denox/memory/container/small_vector.hpp"

static constexpr int64_t DEFAULT_ASSUMPTION = 1920;

denox::SymGraphEval
denox::compiler::assumed_symeval(const SymGraph &symGraph,
                                 memory::span<const NamedValue> valueNames,
                                 const CompileOptions &options) {
  memory::small_vector<SymSpec, 4> symSpecs;
  for (const auto &assumption : options.assumptions.valueAssumptions) {
    auto it =
        std::ranges::find_if(valueNames, [&](const NamedValue &namedValue) {
          return namedValue.name == assumption.name;
        });
    if (it == valueNames.end()) {
      continue;
    }
    auto namedValue = *it;
    if (namedValue.value.isConstant()) {
      continue;
    }
    Sym::symbol symbol = namedValue.value.sym();
    int64_t value = static_cast<int64_t>(assumption.value);
    symSpecs.push_back(SymSpec{.symbol = symbol, .value = value});
  }
  return symGraph.eval(symSpecs, DEFAULT_ASSUMPTION);
}
