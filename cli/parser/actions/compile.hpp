#pragma once

#include "denox/compiler.hpp"
#include "io/IOEndpoint.hpp"
#include "parser/artefact.hpp"
#include <optional>

struct CompileAction {
  OnnxArtefact input; // must exist
  IOEndpoint output;

  // compile options.
  std::optional<DbArtefact> database;
  denox::CompileOptions options;
};
