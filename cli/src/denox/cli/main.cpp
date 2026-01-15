#include "alloc/monotone_alloc.hpp"
#include "denox/cli/compile.hpp"
#include "denox/cli/infer.hpp"
#include "denox/cli/parser/parse.hpp"
#include "denox/cli/populate.hpp"
#include "denox/cli/bench.hpp"
#include "denox/diag/not_implemented.hpp"
#include "denox/io/fs/Path.hpp"
#include <fmt/base.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>

int main(int argc, char **argv) {

  fmt::println("home = {}", denox::io::Path::assets());

  auto action = parse_argv(argc, argv);

  switch (action.kind()) {
  case ActionKind::Compile:
    compile(action.compile());
    break;
  case ActionKind::Infer:
    infer(action.infer());
    break;
  case ActionKind::Bench:
    bench(action.bench());
    break;
  case ActionKind::Populate:
    populate(action.populate());
    break;
  case ActionKind::Help:
    denox::diag::not_implemented();
  case ActionKind::Version:
    denox::diag::not_implemented();
    break;
  }

  mono_free_all();
}
