#pragma once

#include "denox/memory/container/string.hpp"
#include "denox/memory/container/string_view.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

namespace denox::diag {

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
      auto sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
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
      auto sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
      lg = std::make_shared<spdlog::logger>("denox.raw", sink);
      lg->set_level(spdlog::level::trace);
      lg->set_pattern("%v");
      spdlog::register_logger(lg);
    }
    return *lg;
  }();
  return ref;
}

#define DENOX_TRACE(...) ::denox::diag::denox_logger().trace(__VA_ARGS__)
#define DENOX_DEBUG(...) ::denox::diag::denox_logger().debug(__VA_ARGS__)
#define DENOX_INFO(...) ::denox::diag::denox_logger().info(__VA_ARGS__)
#define DENOX_WARN(...) ::denox::diag::denox_logger().warn(__VA_ARGS__)
#define DENOX_ERROR(...) ::denox::diag::denox_logger().error(__VA_ARGS__)

#define DENOX_TRACE_RAW(...)                                                   \
  ::denox::diag::denox_raw_logger().trace(__VA_ARGS__)
#define DENOX_DEBUG_RAW(...)                                                   \
  ::denox::diag::denox_raw_logger().debug(__VA_ARGS__)
#define DENOX_INFO_RAW(...) ::denox::diag::denox_raw_logger().info(__VA_ARGS__)
#define DENOX_WARN_RAW(...) ::denox::diag::denox_raw_logger().warn(__VA_ARGS__)
#define DENOX_ERROR_RAW(...)                                                   \
  ::denox::diag::denox_raw_logger().error(__VA_ARGS__)

class Logger {
private:
  struct AniCodes {
    memory::string_view red = "";
    memory::string_view green = "";
    memory::string_view yellow = "";
    memory::string_view blue = "";
    memory::string_view bold = "";
    memory::string_view reset = "";
  };

public:
  Logger(const memory::string &name, bool colors)
      : m_codes(std::make_shared<AniCodes>()),
        m_sink(std::make_shared<spdlog::sinks::stderr_sink_st>()),
        m_logger(std::make_shared<spdlog::logger>(name, m_sink)) {
    if (colors) {
      m_codes->red = "\x1B[31m";
      m_codes->green = "\x1B[32m";
      m_codes->yellow = "\x1B[33m";
      m_codes->blue = "\x1B[34m";
      m_codes->bold = "\x1B[1m";
      m_codes->reset = "\x1B[0m";
    }
    m_logger->set_pattern("%v");
  }

  memory::string_view red() const { return m_codes->red; }
  memory::string_view green() const { return m_codes->green; }
  memory::string_view yellow() const { return m_codes->yellow; }
  memory::string_view blue() const { return m_codes->blue; }
  memory::string_view bold() const { return m_codes->bold; }
  memory::string_view reset() const { return m_codes->reset; }

  template <typename... Args>
  void trace(spdlog::format_string_t<Args...> fmt, Args &&...args) {
    m_logger->trace(fmt, std::forward<Args>(args)...);
  }

  template <typename T> void trace(const T &msg) { m_logger->trace(msg); }

  template <typename... Args>
  void debug(spdlog::format_string_t<Args...> fmt, Args &&...args) {
    m_logger->debug(fmt, std::forward<Args>(args)...);
  }

  template <typename T> void debug(const T &msg) { m_logger->debug(msg); }

  template <typename... Args>
  void info(spdlog::format_string_t<Args...> fmt, Args &&...args) {
    m_logger->info(fmt, std::forward<Args>(args)...);
  }

  template <typename T> void info(const T &msg) { m_logger->info(msg); }

  template <typename... Args>
  void warn(spdlog::format_string_t<Args...> fmt, Args &&...args) {
    m_logger->warn(fmt, std::forward<Args>(args)...);
  }

  template <typename T> void warn(const T &msg) { m_logger->warn(msg); }

  template <typename... Args>
  void error(spdlog::format_string_t<Args...> fmt, Args &&...args) {
    m_logger->error(fmt, std::forward<Args>(args)...);
  }

  template <typename T> void error(const T &msg) { m_logger->error(msg); }

  // private:
  std::shared_ptr<AniCodes> m_codes;
  std::shared_ptr<spdlog::sinks::stderr_sink_st> m_sink;
  std::shared_ptr<spdlog::logger> m_logger;
};

} // namespace denox::diag
