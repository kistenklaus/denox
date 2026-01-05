#pragma once

#include "denox/cli/parser/artefact.hpp"
#include "denox/compiler/Options.hpp"

struct PopulateAction  {
  OnnxArtefact model;
  DbArtefact database;

  // compile options
  denox::compiler::CompileOptions options;
};
