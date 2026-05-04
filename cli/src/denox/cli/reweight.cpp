#include "denox/cli/reweight.hpp"
#include "denox/cli/io/IOEndpoint.hpp"

void reweight(ReweightAction &action) {
  DnxArtefact &ref = action.reference_artefact;
  OnnxArtefact &model = action.reference_model;
  IOEndpoint &output = action.output;

  fmt::println("no yet implemented");
}
