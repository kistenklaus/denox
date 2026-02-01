#pragma once

#include "denox/cli/io/IOEndpoint.hpp"
#include "denox/cli/parser/artefact.hpp"
struct DumpCsvAction {
  DbArtefact database;
  IOEndpoint csv;
};

