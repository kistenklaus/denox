#include "denox/cli/bench.hpp"
#include "denox/compiler/compile.hpp"
#include "denox/device_info/query/query_driver_device_info.hpp"
#include "denox/runtime/db.hpp"
#include "denox/runtime/instance.hpp"
#include "denox/runtime/model.hpp"
#include <stdexcept>
#include <variant>

void bench(BenchAction &action) {

  denox::diag::Logger logger("denox.bench", action.logcolors, action.loglevel);

  switch (action.target.kind()) {
  case ArtefactKind::Onnx: {
    denox::memory::optional<denox::Db> db;
    if (action.database) {
      if (action.database->endpoint.kind() != IOEndpointKind::Path) {
        denox::diag::invalid_state();
      }
      db = denox::Db::open(action.database->endpoint.path());
    }

    const char *deviceName = nullptr;
    if (std::holds_alternative<IOEndpoint>(action.device)) {
      throw std::runtime_error("invalid device");
    } else if (std::holds_alternative<denox::memory::string>(action.device)) {
      deviceName = std::get<denox::memory::string>(action.device).c_str();
    }


    denox::runtime::ContextHandle context =
        denox::runtime::Context::make(deviceName, action.apiVersion, logger);

    action.options.deviceInfo = denox::query_driver_device_info(
        vk::Instance{context->vkInstance()},
        vk::PhysicalDevice{context->vkPhysicalDevice()}, action.apiVersion);

    auto dnxbuf =
        denox::compile(action.target.onnx().data, db, context, action.options, logger);

    auto model = denox::runtime::Model::make(dnxbuf);
    auto instance = denox::runtime::Instance::make(model, action.valueSpecs, logger);
    instance->bench().report(logger);
    break;
  }
  case ArtefactKind::Dnx: {
    const char *deviceName = nullptr;
    if (std::holds_alternative<IOEndpoint>(action.device)) {
      throw std::runtime_error("invalid device");
    } else if (std::holds_alternative<denox::memory::string>(action.device)) {
      deviceName = std::get<denox::memory::string>(action.device).c_str();
    }
    auto ctx = denox::runtime::Context::make(deviceName, action.apiVersion, logger);
    auto model = denox::runtime::Model::make(action.target.dnx().data, ctx);
    auto instance = denox::runtime::Instance::make(model, action.valueSpecs, logger);
    instance->bench().report(logger);
    break;
  }
  case ArtefactKind::Database: {
    const char *deviceName = nullptr;
    if (std::holds_alternative<IOEndpoint>(action.device)) {
      throw std::runtime_error("invalid device");
    } else if (std::holds_alternative<denox::memory::string>(action.device)) {
      deviceName = std::get<denox::memory::string>(action.device).c_str();
    }
    auto db = denox::Db::open(action.target.database().endpoint.path());
    auto ctx = denox::runtime::Context::make(deviceName, action.apiVersion, logger);
    auto rdb = denox::runtime::Db::open(ctx, db);
    rdb->bench(action.benchOptions, {}, logger);
    break;
  }
  }
}
