#pragma once

#include "denox/device_info/DeviceInfo.hpp"
#include "denox/memory/container/hashmap.hpp"
#include "denox/spirv/SpirvBinary.hpp"
#include <fmt/format.h>
#include <functional>
#include <mutex>
#include <spirv-tools/libspirv.h>
#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>
#include <thread>

namespace denox::spirv {

class SpirvTools {
public:
  SpirvTools(const DeviceInfo &deviceInfo);

  bool validate(const SpirvBinary &binary);
  bool optimize(SpirvBinary &binary);

  std::string get_error_msg() const { return m_logger->get(); }

private:
  struct AsyncLogger {
    std::string get() {
      std::lock_guard lck{mutex};
      return log;
    }

    void message_callback(spv_message_level_t level, const char *source,
                          const spv_position_t &p, const char *msg,
                          const char *stage) {
      static const bool use_color = true;
      const char *R = use_color ? "\x1b[0m" : "";
      const char *B = use_color ? "\x1b[1m" : "";
      const char *RED = use_color ? "\x1b[31m" : "";
      const char *YEL = use_color ? "\x1b[33m" : "";
      const char *CYN = use_color ? "\x1b[36m" : "";
      const char *MAG = use_color ? "\x1b[35m" : "";
      const char *GRY = use_color ? "\x1b[90m" : "";

      // Map level → label/color
      const char *label = "message";
      const char *col = GRY;
      switch (level) {
      case SPV_MSG_FATAL:
        label = "fatal error";
        col = RED;
        break;
      case SPV_MSG_INTERNAL_ERROR:
        label = "internal error";
        col = MAG;
        break;
      case SPV_MSG_ERROR:
        label = "error";
        col = RED;
        break;
      case SPV_MSG_WARNING:
        label = "warning";
        col = YEL;
        break;
      case SPV_MSG_INFO:
        label = "info";
        col = CYN;
        break;
      case SPV_MSG_DEBUG:
        label = "debug";
        col = GRY;
        break;
      default:
        label = "message";
        col = GRY;
        break;
      }

      if (!source) {
        source = stage;
      }
      if (!msg) {
        msg = "(no message)";
      }

      // Trim trailing whitespace/newlines (keep it compact like compilers)
      std::string_view m{msg};
      while (!m.empty() && (m.back() == '\n' || m.back() == '\r' ||
                            m.back() == ' ' || m.back() == '\t')) {
        m.remove_suffix(1);
      }

      std::lock_guard lck{this->mutex};

      // <source>:<line>:<col>: <severity>: <message>
      fmt::format_to(std::back_inserter(log), "{}{}{}", B, source, R);
      if (p.line || p.column)
        fmt::format_to(std::back_inserter(log), ":{}:{}", p.line ? p.line : 0,
                       p.column ? p.column : 0);
      fmt::format_to(std::back_inserter(log), ": {}{}{}: {}\n", col, label, R,
                     m);

      // Optional extra context (word index) in dim text
      if (p.index)
        fmt::format_to(std::back_inserter(log), "{}  note: word-index {}{}\n",
                       GRY, p.index, R);
    }
    std::mutex mutex;
    std::string log;
  };

  struct SpirvToolsThreadState {
    spvtools::SpirvTools tools;
    spvtools::Optimizer optimizer;
    std::shared_ptr<AsyncLogger> logger;
    const char *stage = nullptr;

    SpirvToolsThreadState(spv_target_env env,
                          std::shared_ptr<AsyncLogger> logger)
        : tools(env), optimizer(env) {
      auto messenger = [logger = std::move(logger),
                        this](spv_message_level_t level, const char *source,
                              const spv_position_t &p, const char *msg) {
        logger->message_callback(level, source, p, msg, stage);
      };
      tools.SetMessageConsumer(messenger);
      optimizer.SetMessageConsumer(messenger);
      optimizer.SetTargetEnv(env);
      optimizer.SetValidateAfterAll(true);
      optimizer.RegisterPerformancePasses(true);
    }
  };

  std::shared_ptr<SpirvToolsThreadState> thread_local_state() {
    std::lock_guard lck{m_stateMutex};

    auto it = m_state.find(std::this_thread::get_id());
    if (it == m_state.end()) {
      it = m_state
               .emplace(std::this_thread::get_id(),
                        std::make_shared<SpirvToolsThreadState>(m_targetEnv,
                                                                m_logger))
               .first;
    }
    return it->second;
  }

private:
  spv_target_env m_targetEnv;

  std::shared_ptr<AsyncLogger> m_logger;

  std::mutex m_stateMutex;
  memory::hash_map<std::thread::id, std::shared_ptr<SpirvToolsThreadState>>
      m_state;
  const char *m_current_stage = nullptr;
};

} // namespace denox::spirv
