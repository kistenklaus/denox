#pragma once

#include "db/Db.hpp"
#include "heuristic/IHeuristic.hpp"
#include <cmath>
#include <limits>

namespace denox::compiler {

class DbHeuristic : public IHeuristic {
public:
  DbHeuristic(Db db) : m_db(std::move(db)) {}

  float
  eval([[maybe_unused]] std::span<const TensorInstance *> in,
       [[maybe_unused]] const TensorInstance &out, unsigned int pattern,
       unsigned int config, uint64_t hash,
       [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance,
                                                         ComputeOp> &match,
       const IShader *shader) const final override {
    std::string name = shader->name(pattern, config);
    auto db = m_db.get();

    for (const auto &op : db->operations) {
      if (op.pattern != pattern) {
        continue;
      }
      if (op.config != config) {
        continue;
      }
      if (op.shaderName != name) {
        continue;
      }
      if (op.hash != hash) {
        continue;
      }

      bool incomplete = false;
      uint64_t latency_ns = 0;
      for (unsigned int i = 0; i < op.dispatches.size(); ++i) {
        if (db->dispatches[op.dispatches[i]].time.samples > 0) {
          latency_ns += db->dispatches[op.dispatches[i]].time.latency_ns;
        } else {
          incomplete = true;
          break;
        }
      }
      if (incomplete) {
        continue;
      } else {
        return static_cast<float>(latency_ns) * 1e-6f;
      }
    }
    return std::numeric_limits<float>::infinity();
  }

  memory::string weight_to_string(float weight) const final override {
    if (std::isinf(weight)) {
      return fmt::format("unknown");
    } else {
      return fmt::format("{}ms", weight);
    }
  }

private:
  Db m_db;
};

} // namespace denox::compiler
