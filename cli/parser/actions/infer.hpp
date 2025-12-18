#pragma once

#include "denox/compiler.hpp"
#include "parser/artefact.hpp"
#include <optional>

struct InferAction {
  Artefact model;
  IOEndpoint input;
  IOEndpoint output;

  // compile options:
  std::optional<DbArtefact> database;
  std::optional<denox::CompileOptions> options;
};
