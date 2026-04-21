#include "denox/cli/compile.hpp"
#include "denox/compiler/compile.hpp"
#include "denox/device_info/query/query_driver_device_info.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/io/fs/File.hpp"
#include "denox/runtime/context.hpp"
#include <vulkan/vulkan.hpp>

void compile(CompileAction &action) {
  const char *deviceName = nullptr;
  if (action.deviceName.has_value()) {
    deviceName = action.deviceName->c_str();
  }
  denox::runtime::ContextHandle context =
      denox::runtime::Context::make(deviceName, action.apiVersion);

  action.options.deviceInfo = denox::query_driver_device_info(
      vk::Instance{context->vkInstance()},
      vk::PhysicalDevice{context->vkPhysicalDevice()}, action.apiVersion);

  denox::memory::optional<denox::Db> db;
  if (action.database) {
    if (action.database->endpoint.kind() != IOEndpointKind::Path) {
      denox::diag::invalid_state();
    }
    db = denox::Db::open(action.database->endpoint.path());
  }

  auto dnxbuf = denox::compile(action.input.data, db, context, action.options);

  switch (action.output.kind()) {
  case IOEndpointKind::Path: {
    auto outfile = denox::io::File::open(
        action.output.path(), denox::io::File::OpenMode::Create |
                                  denox::io::File::OpenMode::Write |
                                  denox::io::File::OpenMode::Truncate);
    outfile.write_exact(dnxbuf);
    break;
  }
  case IOEndpointKind::Pipe:
    Pipe{}.write_exact(dnxbuf);
  }

  if (db.has_value()) {
    db->checkpoint();
  }
}
