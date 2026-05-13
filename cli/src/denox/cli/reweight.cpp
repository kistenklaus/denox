#include "denox/cli/reweight.hpp"
#include "denox/cli/io/IOEndpoint.hpp"
#include "denox/cli/io/OutputStream.hpp"
#include "denox/compiler/reweight.hpp"
#include "denox/memory/container/vector.hpp"

void reweight(ReweightAction &action) {
  DnxArtefact &ref = action.reference_artefact;
  OnnxArtefact &model = action.reference_model;
  IOEndpoint &output = action.output;

  using namespace denox;

  memory::vector<std::byte> dnx{ref.data.begin(), ref.data.end()};

  diag::Logger logger("denox.reweight", action.logcolors, action.loglevel);

  denox::reweight(dnx, model.data, logger);

  OutputStream out(output);
  out.write_exact(dnx);
}
