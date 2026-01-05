#pragma once

#include "denox/cli/parser/artefact.hpp"
#include "denox/compiler/Options.hpp"
#include <optional>

struct InferAction {
  Artefact model;
  IOEndpoint input;
  IOEndpoint output;

  // compile options:
  std::optional<DbArtefact> database;
  std::optional<denox::compiler::CompileOptions> options;
};
