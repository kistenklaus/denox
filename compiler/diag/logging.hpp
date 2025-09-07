#pragma once

namespace denox::compiler::diag {

enum LogLevel {
  Info,
  Debug,
  Trace,
};

#define DENOX_TRACE(...) SPDLOG_LOGGER_TRACE(spdlog::get("denox"), __VA_ARGS__)
#define DENOX_DEBUG(...) SPDLOG_LOGGER_DEBUG(spdlog::get("denox"), __VA_ARGS__)
#define DENOX_INFO(...) SPDLOG_LOGGER_INFO(spdlog::get("denox"), __VA_ARGS__)
#define DENOX_WARN(...) SPDLOG_LOGGER_WARN(spdlog::get("denox"), __VA_ARGS__)
#define DENOX_ERROR(...) SPDLOG_LOGGER_ERROR(spdlog::get("denox"), __VA_ARGS__)
#define DENOX_TRACE_RAW(...) SPDLOG_LOGGER_TRACE(spdlog::get("denox.raw"), __VA_ARGS__)
#define DENOX_DEBUG_RAW(...) SPDLOG_LOGGER_DEBUG(spdlog::get("denox.raw"), __VA_ARGS__)
#define DENOX_INFO_RAW(...) SPDLOG_LOGGER_INFO(spdlog::get("denox.raw"), __VA_ARGS__)
#define DENOX_WARN_RAW(...) SPDLOG_LOGGER_WARN(spdlog::get("denox.raw"), __VA_ARGS__)
#define DENOX_ERROR_RAW(...) SPDLOG_LOGGER_ERROR(spdlog::get("denox.raw"), __VA_ARGS__)

void initLogger();

}
