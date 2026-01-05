#pragma once

#include "denox/db/Db.hpp"
#include "denox/device_info/ApiVersion.hpp"
#include "denox/runtime/context.hpp"

namespace denox::runtime {

struct DbBenchOptions {
  uint32_t minSamples = 10;
  float maxRelativeError = 0.05f;
  bool saveProgress = true;
};

class Db {
public:
  static std::shared_ptr<Db> open(const io::Path &path) {
    auto context = Context::make(nullptr, ApiVersion::VULKAN_1_4);
    return open(context, path);
  }

  static std::shared_ptr<Db> open(const denox::Db &db) {
    auto context = Context::make(nullptr, ApiVersion::VULKAN_1_4);
    return open(context, db);
  }

  static std::shared_ptr<Db> open(const ContextHandle &context,
                                  const io::Path &path) {
    auto db = denox::Db::open(path);
    return open(context, db);
  }

  static std::shared_ptr<Db> open(const ContextHandle &context, denox::Db db) {
    return std::shared_ptr<Db>(new Db(context, db));
  }

  void bench(const DbBenchOptions &options = {});

private:
  explicit Db(const ContextHandle &context, const denox::Db &db)
      : m_context(context), m_db(db) {}

  ContextHandle m_context;
  denox::Db m_db;
};

using DbHandle = std::shared_ptr<Db>;

} // namespace denox::runtime
