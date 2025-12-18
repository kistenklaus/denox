#pragma once

#include <cassert>
#include <charconv>
#include <cstdint>
#include <optional>
#include <string_view>

class Duration {
public:
  constexpr Duration() noexcept = default;

  static std::optional<Duration> parse(std::string_view sv) noexcept {
    if (sv.empty()) {
      return std::nullopt;
    }
    std::size_t i = 0;
    while (i < sv.size() && is_digit(sv[i])) {
      ++i;
    }

    if (i == 0 || i == sv.size()) {
      return std::nullopt;
    }

    std::uint64_t value{};
    auto res = std::from_chars(sv.data(), sv.data() + i, value);
    if (res.ec != std::errc{} || res.ptr != sv.data() + i) {
      return std::nullopt;
    }

    std::string_view unit_str = sv.substr(i);
    Unit unit;

    if (unit_str == "ns") {
      unit = Unit::Nanoseconds;
    } else if (unit_str == "us") {
      unit = Unit::Microseconds;
    } else if (unit_str == "ms") {
      unit = Unit::Milliseconds;
    } else if (unit_str == "s") {
      unit = Unit::Seconds;
    } else if (unit_str == "min") {
      unit = Unit::Minutes;
    } else if (unit_str == "h") {
      unit = Unit::Hours;
    } else {
      return std::nullopt;
    }

    return Duration(value, unit);
  }

  std::uint64_t nanoseconds() const noexcept { return m_nanoseconds; }

  std::uint64_t milliseconds() const noexcept {
    return m_nanoseconds / 1'000'000;
  }

  std::uint64_t seconds() const noexcept {
    return m_nanoseconds / 1'000'000'000;
  }

private:
  enum class Unit {
    Nanoseconds,
    Microseconds,
    Milliseconds,
    Seconds,
    Minutes,
    Hours,
  };
  static constexpr bool is_digit(char c) noexcept {
    return c >= '0' && c <= '9';
  }

  static constexpr std::uint64_t scale(Unit u) noexcept {
    switch (u) {
    case Unit::Nanoseconds:
      return 1;
    case Unit::Microseconds:
      return 1'000;
    case Unit::Milliseconds:
      return 1'000'000;
    case Unit::Seconds:
      return 1'000'000'000;
    case Unit::Minutes:
      return 60ull * 1'000'000'000;
    case Unit::Hours:
      return 60ull * 60ull * 1'000'000'000;
    }
    return 1;
  }

  Duration(std::uint64_t value, Unit unit)
      : m_nanoseconds(value * scale(unit)) {}

private:
  std::uint64_t m_nanoseconds = 0;
};
