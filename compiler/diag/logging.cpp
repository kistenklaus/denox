#include "diag/logging.hpp"
#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace denox::compiler::diag {



void initLogger() {
  if (!spdlog::get("denox")) {
    auto logger = std::make_shared<spdlog::logger>(
        "denox", std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    logger->set_level(spdlog::level::trace);
    if (true) {
      logger->set_pattern("[%^%-5l%$ \x1B[4m%s:%#\x1B[0m] %v");
      logger->flush_on(spdlog::level::trace);
    } else {
      logger->set_pattern("[%^%l%$]: %v");
      logger->flush_on(spdlog::level::info);
    }
    spdlog::register_logger(logger);
  }

  if (!spdlog::get("denox.raw")) {
    auto raw = std::make_shared<spdlog::logger>(
        "denox.raw", std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    raw->set_level(spdlog::level::trace);
    if (true) {
      raw->set_pattern("%v");
    } 
    spdlog::register_logger(raw);
  }
}

} // namespace denox::compiler::diag
