#include "denox/cli/compile.hpp"
#include "denox/cli/device_yml/device_yml.hpp"
#include "denox/cli/io/InputStream.hpp"
#include "denox/compiler/compile.hpp"
#include "denox/device_info/query/query_driver_device_info.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/io/fs/File.hpp"
#include "denox/runtime/context.hpp"
#include <vulkan/vulkan.hpp>

void compile(CompileAction &action) {

  denox::diag::Logger logger("denox.compile", action.logcolors,
                             action.loglevel);

  denox::runtime::ContextHandle context;

  if (std::holds_alternative<IOEndpoint>(action.device)) {
    InputStream istream{std::get<IOEndpoint>(action.device)};
    denox::memory::vector<std::byte> yml(1 << 20);
    size_t sz = istream.read(yml);
    action.options.deviceInfo =
        deserialize_device_yml(denox::memory::span{yml.data(), sz});
    const char *deviceName = action.options.deviceInfo.name.c_str();
    context = denox::runtime::Context::make(
        deviceName, action.options.deviceInfo.apiVersion);
  } else if (std::holds_alternative<denox::memory::string>(action.device)) {
    const char *deviceName =
        std::get<denox::memory::string>(action.device).c_str();
    context =
        denox::runtime::Context::make(deviceName, action.apiVersion, logger);
    action.options.deviceInfo = denox::query_driver_device_info(
        vk::Instance{context->vkInstance()},
        vk::PhysicalDevice{context->vkPhysicalDevice()}, action.apiVersion);
  } else {
    context = denox::runtime::Context::make(nullptr, action.apiVersion, logger);
    action.options.deviceInfo = denox::query_driver_device_info(
        vk::Instance{context->vkInstance()},
        vk::PhysicalDevice{context->vkPhysicalDevice()}, action.apiVersion);
  }

  denox::memory::optional<denox::Db> db;
  if (action.database) {
    if (action.database->endpoint.kind() != IOEndpointKind::Path) {
      denox::diag::invalid_state();
    }
    db = denox::Db::open(action.database->endpoint.path());
  }

  auto dnxbuf =
      denox::compile(action.input.data, db, context, action.options, logger);

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
