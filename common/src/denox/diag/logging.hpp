#pragma once

#include "denox/io/is_tty.hpp"
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

class Logger {
private:
  struct AniCodes {
    memory::string_view red = "";
    memory::string_view green = "";
    memory::string_view yellow = "";
    memory::string_view blue = "";
    memory::string_view gray = "";
    memory::string_view bold = "";
    memory::string_view reset = "";
    memory::string_view clear_line = "";
    memory::string_view cursor_up = "";
  };

public:
  Logger(const memory::string &name, bool colors)
      : m_codes(std::make_shared<AniCodes>()),
        m_sink(std::make_shared<spdlog::sinks::stderr_sink_st>()),
        m_logger(std::make_shared<spdlog::logger>(name, m_sink)) {

    if (!denox::io::stderr_is_tty()) {
      colors = false;
    }
    if (colors) {
      m_codes->red = "\x1B[31m";
      m_codes->green = "\x1B[32m";
      m_codes->yellow = "\x1B[33m";
      m_codes->blue = "\x1B[34m";
      m_codes->gray = "\x1B[90m";
      m_codes->bold = "\x1B[1m";
      m_codes->reset = "\x1B[0m";
      m_codes->clear_line = "\r\x1B[K";
      m_codes->cursor_up = "\x1B[A";
    }
    m_logger->set_pattern("%v");
  }

  memory::string_view red() const { return m_codes->red; }
  memory::string_view green() const { return m_codes->green; }
  memory::string_view yellow() const { return m_codes->yellow; }
  memory::string_view blue() const { return m_codes->blue; }
  memory::string_view gray() const { return m_codes->gray; }
  memory::string_view bold() const { return m_codes->bold; }
  memory::string_view reset() const { return m_codes->reset; }

  memory::string_view clear_line() const { return m_codes->clear_line; }
  memory::string_view cursor_up() const { return m_codes->cursor_up; }

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
    m_logger->warn(fmt::format(fmt, std::forward<Args>(args)...));
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
