#pragma once

#include "denox/cli/parser/artefact.hpp"
#include "denox/diag/logging.hpp"
#include "denox/memory/container/optional.hpp"

struct ReweightAction {
  DnxArtefact reference_artefact;
  OnnxArtefact reference_model;
  IOEndpoint output;

  denox::diag::LogLevel loglevel = denox::diag::LogLevel::Info;
  bool logcolors = true;
};
