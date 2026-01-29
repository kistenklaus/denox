#include "denox/cli/bench.hpp"
#include "denox/compiler/compile.hpp"
#include "denox/device_info/query/query_driver_device_info.hpp"
#include "denox/diag/not_implemented.hpp"
#include "denox/runtime/db.hpp"
#include "denox/runtime/instance.hpp"
#include "denox/runtime/model.hpp"

void bench(BenchAction &action) {
  switch (action.target.kind()) {
  case ArtefactKind::Onnx: {
    denox::memory::optional<denox::Db> db;
    if (action.database) {
      if (action.database->endpoint.kind() != IOEndpointKind::Path) {
        denox::diag::invalid_state();
      }
      db = denox::Db::open(action.database->endpoint.path());
    }

    action.options.deviceInfo =
        denox::query_driver_device_info(action.apiVersion, action.deviceName);

    auto dnxbuf = denox::compile(action.target.onnx().data, db, action.options);

    const char *device = nullptr;
    if (action.deviceName) {
      device = action.deviceName->c_str();
    }

    auto ctx = denox::runtime::Context::make(device, action.apiVersion);
    auto model = denox::runtime::Model::make(dnxbuf);
    auto instance = denox::runtime::Instance::make(model, action.valueSpecs);
    instance->bench().report();
    break;
  }
  case ArtefactKind::Dnx: {
    const char *device = nullptr;
    if (action.deviceName) {
      device = action.deviceName->c_str();
    }
    auto ctx = denox::runtime::Context::make(device, action.apiVersion);
    auto model = denox::runtime::Model::make(action.target.dnx().data);
    auto instance = denox::runtime::Instance::make(model, action.valueSpecs);
    instance->bench().report();
    break;
  }
  case ArtefactKind::Database: {
    const char *device = nullptr;
    if (action.deviceName) {
      device = action.deviceName->c_str();
    }
    auto ctx = denox::runtime::Context::make(device, action.apiVersion);
    auto db = denox::Db::open(action.target.database().endpoint.path());
    auto rdb = denox::runtime::Db::open(ctx, db);
    rdb->bench(action.benchOptions);
    break;
  }
  }
}
