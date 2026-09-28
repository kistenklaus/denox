#pragma once

#include "denox/compiler/dce/ConstModel.hpp"
#include "denox/compiler/implement/Supergraph.hpp"
#include <stdexcept>

namespace denox::compiler {

struct FailedToImplement : public std::runtime_error {
  explicit FailedToImplement(std::string msg) noexcept
      : std::runtime_error{fmt::format(
            "Failed to implement at least one of the following operations:\n{}",
            msg)} {}
};

[[noreturn]] void failed_to_implement(const SuperGraph &supergraph,
                                      const ConstModel &model);
} // namespace denox::compiler
