

#include "denox/cli/io/IOEndpoint.hpp"
#include "denox/cli/parser/artefact.hpp"
#include "denox/diag/logging.hpp"
struct DumpAction {
  DnxArtefact dnx;
  IOEndpoint output = IOEndpoint(Pipe());

  denox::diag::LogLevel loglevel = denox::diag::LogLevel::Info;
  bool logcolors = true;
};
