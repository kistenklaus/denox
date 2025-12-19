

#include "alloc/monotone_alloc.hpp"
#include "compile.hpp"
#include "denox/compiler.h"
#include "parser/parse.hpp"
#include <fmt/printf.h>
#include <stdexcept>

int main(int argc, char **argv) {

  DenoxDeviceInfo deviceInfo;
  if (denox_query_device_info("*RTX*", &deviceInfo) != DENOX_SUCCESS) {
    // throw std::runtime_error("fuck");
  }
  fmt::println("{}", deviceInfo.deviceName);

  denox_destroy_device_info(&deviceInfo);

  return 0;

  std::optional<Action> actionOpt;
  try {
    actionOpt = parse_argv(argc, argv);
  } catch (const std::exception &e) {
    fmt::println("{}", e.what());
    return -1;
  }

  const Action &action = actionOpt.value();
  switch (action.kind()) {
  case ActionKind::Compile: {
    const auto &ac = action.compile();
    compile(ac.input, ac.output, ac.database, ac.options);
    break;
  }
  case ActionKind::Infer:
    break;
  case ActionKind::Bench:
    break;
  case ActionKind::Populate:
    break;
  case ActionKind::Help:
    break;
  case ActionKind::Version:
    break;
  }

  mono_free_all();
}
