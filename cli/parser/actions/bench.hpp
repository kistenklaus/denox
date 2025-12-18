#pragma once

#include "denox/compiler.hpp"
#include "io/IOEndpoint.hpp"
#include "parser/artefact.hpp"
#include <optional>

struct BenchAction {
  Artefact target;

  // compile options.
  std::optional<DbArtefact> database;
  std::optional<denox::CompileOptions> options;

  // bench options.
  std::optional<denox::Shape> inputSize;
};
