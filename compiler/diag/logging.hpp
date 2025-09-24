#pragma once

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace denox::compiler::diag {

enum LogLevel {
  Info,
  Debug,
  Trace,
};

inline spdlog::logger &denox_logger() {
  // thread-safe since C++11 for function-local statics
  static spdlog::logger &ref = []() -> spdlog::logger & {
    auto lg = spdlog::get("denox");
    if (!lg) {
      auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
      lg = std::make_shared<spdlog::logger>("denox", sink);
      lg->set_level(spdlog::level::trace);
      lg->set_pattern("[%^%-5l%$ \x1B[4m%s:%#\x1B[0m] %v");
      lg->flush_on(spdlog::level::trace);
      spdlog::register_logger(lg);
    }
    return *lg;
  }();
  return ref;
}

inline spdlog::logger &denox_raw_logger() {
  static spdlog::logger &ref = []() -> spdlog::logger & {
    auto lg = spdlog::get("denox.raw");
    if (!lg) {
      auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
      lg = std::make_shared<spdlog::logger>("denox.raw", sink);
      lg->set_level(spdlog::level::trace);
      lg->set_pattern("%v");
      spdlog::register_logger(lg);
    }
    return *lg;
  }();
  return ref;
}

#define DENOX_TRACE(...)                                                       \
  ::denox::compiler::diag::denox_logger().trace(__VA_ARGS__)
#define DENOX_DEBUG(...)                                                       \
  ::denox::compiler::diag::denox_logger().debug(__VA_ARGS__)
#define DENOX_INFO(...)                                                        \
  ::denox::compiler::diag::denox_logger().info(__VA_ARGS__)
#define DENOX_WARN(...)                                                        \
  ::denox::compiler::diag::denox_logger().warn(__VA_ARGS__)
#define DENOX_ERROR(...)                                                       \
  ::denox::compiler::diag::denox_logger().error(__VA_ARGS__)

#define DENOX_TRACE_RAW(...)                                                   \
  ::denox::compiler::diag::denox_raw_logger().trace(__VA_ARGS__)
#define DENOX_DEBUG_RAW(...)                                                   \
  ::denox::compiler::diag::denox_raw_logger().debug(__VA_ARGS__)
#define DENOX_INFO_RAW(...)                                                    \
  ::denox::compiler::diag::denox_raw_logger().info(__VA_ARGS__)
#define DENOX_WARN_RAW(...)                                                    \
  ::denox::compiler::diag::denox_raw_logger().warn(__VA_ARGS__)
#define DENOX_ERROR_RAW(...)                                                   \
  ::denox::compiler::diag::denox_raw_logger().error(__VA_ARGS__)

} // namespace denox::compiler::diag
