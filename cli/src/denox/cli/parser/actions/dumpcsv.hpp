#pragma once

#include "denox/cli/io/IOEndpoint.hpp"
#include "denox/cli/parser/artefact.hpp"
#include "denox/diag/logging.hpp"
struct DumpCsvAction {
  DbArtefact database;
  IOEndpoint csv;
};

