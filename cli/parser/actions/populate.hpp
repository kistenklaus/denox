#pragma once

#include "denox/compiler.hpp"
#include "parser/artefact.hpp"

struct PopulateAction  {
  OnnxArtefact model;
  DbArtefact database;

  // compile options
  denox::CompileOptions options;
};
