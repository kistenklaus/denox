#pragma once

#include "denox/io/is_tty.hpp"
#include "denox/memory/container/string.hpp"
#include "denox/memory/container/string_view.hpp"
#include <fmt/format.h>
#include <memory>

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
  Logger([[maybe_unused]] const memory::string &name, bool colors)
      : m_codes(std::make_shared<AniCodes>()) {
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
  void trace(fmt::format_string<Args...> fmt, Args &&...args) {
    fmt::println(fmt, std::forward<Args>(args)...);
  }

  template <typename T> void trace(const T &msg) { fmt::println("{}", msg); }

  template <typename... Args>
  void debug(fmt::format_string<Args...> fmt, Args &&...args) {
    fmt::println(fmt, std::forward<Args>(args)...);
  }

  template <typename T> void debug(const T &msg) { fmt::println("{}", msg); }

  template <typename... Args>
  void info(fmt::format_string<Args...> fmt, Args &&...args) {
    fmt::println(fmt, std::forward<Args>(args)...);
  }

  template <typename T> void info(const T &msg) { fmt::println("{}", msg); }

  template <typename... Args>
  void warn(fmt::format_string<Args...> fmt, Args &&...args) {
    fmt::println(fmt, std::forward<Args>(args)...);
  }

  template <typename T> void warn(const T &msg) { fmt::println("{}", msg); }

  template <typename... Args>
  void error(fmt::format_string<Args...> fmt, Args &&...args) {
    fmt::println(fmt, std::forward<Args>(args)...);
  }

  template <typename T> void error(const T &msg) { fmt::println("{}", msg); }

  // private:
  std::shared_ptr<AniCodes> m_codes;
};

} // namespace denox::diag
