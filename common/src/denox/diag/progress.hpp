#pragma once

#include "denox/diag/logging.hpp"
namespace denox::diag {

class Progress {
public:
  Progress(float start = 0.0f, float end = 1.0f) : m_start(start), m_end(end) {}

  template <typename... Args>
  void step(Logger &logger, float progress, fmt::format_string<Args...> fmt,
            Args &&...args) {
    float rel = m_start * (1 - progress) + m_end * progress;
    const uint32_t percentage = static_cast<uint32_t>(std::round(rel * 100));
    logger.info(fmt::format("[{:>3}%] {}", percentage,
                            fmt::format(fmt, std::forward<Args>(args)...)));
  }

  template <typename... Args>
  void step_inplace(Logger &logger, float progress, bool skip_clear,
                    fmt::format_string<Args...> fmt, Args &&...args) {
    float rel = m_start * (1 - progress) + m_end * progress;
    const uint32_t percentage = static_cast<uint32_t>(std::round(rel * 100));
    if (skip_clear) {
      logger.info("[{:>3}%] {}", percentage,
                  fmt::format(fmt, std::forward<Args>(args)...));
    } else {
      logger.info("{}{}[{:>3}%] {}", logger.cursor_up(), logger.clear_line(), percentage,
                  fmt::format(fmt, std::forward<Args>(args)...));
    }
  }

  Progress sub_progress(float start, float end) {
    float rstart = m_start * (1.0f - start) + m_end * start;
    float rend = m_start * (1.0f - end) + m_end * end;
    return Progress(rstart, rend);
  }

private:
  float m_start;
  float m_end;
};

} // namespace denox::diag
