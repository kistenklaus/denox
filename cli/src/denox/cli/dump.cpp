#include "dump.hpp"
#include "denox/device_info/CoopmatProperties.hpp"
#include <algorithm>
#include <dnx.h>
#include <set>
#include <sstream>
#include <stdexcept>

static void binary_size(const denox::dnx::Model *dnx, size_t dnx_size,
                        std::stringstream &ss) {
  size_t param_size = 0;
  for (uint32_t i = 0; i < dnx->initializers()->size(); ++i) {
    const auto *init = dnx->initializers()->Get(i);
    param_size += init->data()->size();
  }

  size_t spv_size = 0;
  for (uint32_t i = 0; i < dnx->shader_binaries()->size(); ++i) {
    const auto *binary = dnx->shader_binaries()->Get(i);
    spv_size += binary->spirv()->size() * sizeof(uint32_t);
  }

  {
    ss << fmt::format("Binary-Size-Summary:\n");
    ss << fmt::format("- weights             : {:.2f}MB\n", param_size / 1e6);
    ss << fmt::format("- spirv               : {:.2f}MB\n", spv_size / 1e6);
    ss << fmt::format("- metadata (overhead) : {:.2f}KB\n",
                      (dnx_size - (param_size + spv_size)) / 1e3);
    ss << fmt::format("- total               : {:.2f}MB\n", dnx_size / 1e6);
  }
}

static void model_info(const denox::dnx::Model *dnx, std::stringstream &ss) {
  const auto *info = dnx->info();
  ss << fmt::format("Model-Info:\n");
  if (info && info->producer()) {
    ss << fmt::format("- producer            : {}\n", info->producer()->str());
  } else {
    ss << fmt::format("- producer            : none\n");
  }
  if (info && info->producer_version()) {
    ss << fmt::format("- producer-version    : {}\n", info->producer()->str());
  } else {
    ss << fmt::format("- producer-version    : none\n");
  }
  if (info && info->model_version()) {
    ss << fmt::format("- version             : {}\n",
                      info->model_version()->str());
  } else {
    ss << fmt::format("- version             : none\n");
  }
}

static void compilation_info(const denox::dnx::Model *dnx,
                             std::stringstream &ss) {
  ss << fmt::format("Compilation-Info:\n");

  denox::dnx::Version version = dnx->version();
  switch (version) {
  case denox::dnx::Version_DNX_VERSION_1_0:
    ss << fmt::format("- dnx-version        : 1.0\n");
    break;
  default:
    ss << fmt::format("- dnx-version        : INVALID\n");
    break;
  }

  const auto *model_info = dnx->info();

  const denox::dnx::CompilationInfo *info =
      model_info ? model_info->compilation_info() : nullptr;

  if (info && info->denox_commit_hash()) {
    ss << fmt::format("- denox-commit       : {}\n",
                      info->denox_commit_hash()->str());
  } else {
    ss << fmt::format("- denox-commit       : ?\n");
  }

  if (info) {
    ss << fmt::format("- optimization-level : {}\n",
                      info->optimization_level());
  } else {
    ss << fmt::format("- optimization-level : ?\n");
  }

  const denox::dnx::CompilationFeatures *features =
      info ? info->features() : nullptr;
  if (features) {
    ss << fmt::format("- fcoopmat           : {}\n", features->coopmat());
  } else {
    ss << fmt::format("- fcoopmat           : ?\n");
  }

  if (features) {
    ss << fmt::format("- fconcat-conv       : {}\n",
                      features->concat_conv_fusion());
  } else {
    ss << fmt::format("- fconcat-conv       : ?\n");
  }

  if (features) {
    ss << fmt::format("- fconv-pool         : {}\n",
                      features->conv_pool_fusion());
  } else {
    ss << fmt::format("- fconv-pool         : ?\n");
  }
  if (features) {
    ss << fmt::format("- fconv-relu         : {}\n",
                      features->conv_relu_fusion());
  } else {
    ss << fmt::format("- fconv-relu         : ?\n");
  }

  if (features) {
    ss << fmt::format("- fupsample-conv     : {}\n",
                      features->upsample_conv_fusion());
  } else {
    ss << fmt::format("- fupsample-conv     : ?\n");
  }
  if (features) {
    ss << fmt::format("- fmemcat            : {}\n",
                      features->implicit_concat());
  } else {
    ss << fmt::format("- fmemcat            : ?\n");
  }

  if (info && info->descriptor_policy()) {
    const auto *policy = info->descriptor_policy();
    ss << fmt::format("- descriptor-policy  : [{},{},{},{},{}]\n",
                      policy->input_policy(), policy->output_policy(),
                      policy->param_policy(), policy->read_policy(),
                      policy->write_policy());
  } else {
    ss << fmt::format("- descriptor-policy  : ?\n");
  }

  const auto *device_info = info ? info->device_info() : nullptr;

  if (device_info && device_info->device_name()) {
    ss << fmt::format("- device-name        : {}\n",
                      device_info->device_name()->str());
  } else {
    ss << fmt::format("- device-name        : ?\n");
  }
  if (device_info) {
    switch (device_info->api_version()) {
    case denox::dnx::VulkanApiVersion_VULKAN_1_0:
      ss << fmt::format("- api-version        : vulkan1.0\n");
      break;
    case denox::dnx::VulkanApiVersion_VULKAN_1_1:
      ss << fmt::format("- api-version        : vulkan1.1\n");
      break;
    case denox::dnx::VulkanApiVersion_VULKAN_1_2:
      ss << fmt::format("- api-version        : vulkan1.2\n");
      break;
    case denox::dnx::VulkanApiVersion_VULKAN_1_3:
      ss << fmt::format("- api-version        : vulkan1.3\n");
      break;
    case denox::dnx::VulkanApiVersion_VULKAN_1_4:
      ss << fmt::format("- api-version        : vulkan1.4\n");
      break;
    }
  } else {
    ss << fmt::format("- api-version        : ?\n");
  }
  if (device_info) {
    ss << fmt::format("- max-workgroup-count: [{},{},{}]\n",
                      device_info->max_compute_workgroup_count_x(),
                      device_info->max_compute_workgroup_count_y(),
                      device_info->max_compute_workgroup_count_z());
  } else {
    ss << fmt::format("- max-workgroup-count: ?\n");
  }

  if (device_info) {
    ss << fmt::format("- max-workgroup-size : [{},{},{}]\n",
                      device_info->max_compute_workgroup_size_x(),
                      device_info->max_compute_workgroup_size_y(),
                      device_info->max_compute_workgroup_size_z());
  } else {
    ss << fmt::format("- max-workgroup-size : ?\n");
  }

  if (device_info) {
    ss << fmt::format("- max-workgroup-invoc: {}\n",
                      device_info->max_compute_workgroup_invocations());
  } else {
    ss << fmt::format("- max-workgroup-invoc: ?\n");
  }

  if (device_info && device_info->max_compute_workgroup_subgroups()) {
    ss << fmt::format("- max-sg-per-wg      : {}\n",
                      device_info->max_compute_workgroup_subgroups());
  } else {
    ss << fmt::format("- max-sg-per-wg      : ?\n");
  }
  if (device_info) {
    ss << fmt::format("- max-shared-memory  : {}\n",
                      device_info->max_compute_shared_memory());
  } else {
    ss << fmt::format("- max-shared-memory  : ?\n");
  }
  if (device_info) {
    ss << fmt::format("- max-pc-size        : {}\n",
                      device_info->max_push_constant_size());
  } else {
    ss << fmt::format("- max-pc-size        : ?\n");
  }
}

static void dispatch(const denox::dnx::Model *dnx, uint32_t i,
                     std::stringstream &ss) {
  const auto *dispatch = dnx->dispatches()->Get(i);
  const auto *info = dispatch->info();
  std::string name;
  if (info && info->name()) {
    ss << fmt::format("[{}]: {}\n", i, info->name()->str());
  } else {
    ss << fmt::format("[{}]:\n", i);
  }
  if (info && info->debug_info()) {
    ss << fmt::format("- debug-name          : {}\n",
                      info->debug_info()->str());
  }
  if (info && info->src_path()) {
    ss << fmt::format("- shader-source       : {}\n", info->src_path()->str());
  }

  {
    ss << fmt::format("- binary              : [{}] -> {:.1f}KB\n",
                      dispatch->binary_id(),
                      static_cast<float>(dnx->shader_binaries()
                                             ->Get(dispatch->binary_id())
                                             ->spirv()
                                             ->size() *
                                         sizeof(uint32_t)) /
                          1e3f);
  }
  if (dispatch->entry_point()) {
    ss << fmt::format("- entry-point         : {}\n",
                      dispatch->entry_point()->str());
  } else {
    ss << fmt::format("- entry-point         : main\n");
  }
  const denox::dnx::ComputeDispatchRequirements *req = dispatch->requirements();
  if (req && req->fixed_subgroup_size()) {
    ss << fmt::format("- required-sg-control : {}\n",
                      req->fixed_subgroup_size());
  } else {
    ss << fmt::format("- required-sg-size    : none\n");
  }

  if (req && req->required_coopmat_shapes()) {
    const auto *coopmats = req->required_coopmat_shapes();
    std::string shapes = "[";
    for (uint32_t i = 0; i < coopmats->size(); ++i) {
      const auto *coopmat = coopmats->Get(i);
      if (i != 0) {
        shapes += ", ";
      }
      std::string type;
      assert(coopmat->atype() == coopmat->btype() &&
             coopmat->atype() == coopmat->ctype() &&
             coopmat->atype() == coopmat->acctype());
      switch (coopmat->atype()) {
      case denox::dnx::ScalarType_I16:
        type = "i16";
        break;
      case denox::dnx::ScalarType_U16:
        type = "u16";
        break;
      case denox::dnx::ScalarType_I32:
        type = "i32";
        break;
      case denox::dnx::ScalarType_U32:
        type = "u32";
        break;
      case denox::dnx::ScalarType_I64:
        type = "i64";
        break;
      case denox::dnx::ScalarType_U64:
        type = "u64";
        break;
      case denox::dnx::ScalarType_F16:
        type = "f16";
        break;
      case denox::dnx::ScalarType_F32:
        type = "f32";
        break;
      case denox::dnx::ScalarType_F64:
        type = "f64";
        break;
      }

      shapes += fmt::format("{}x{}x{}{}", coopmat->m(), coopmat->k(),
                            coopmat->n(), type);
    }

    shapes += "]";

    ss << fmt::format("- required-coopmat    : {}\n", shapes);
  } else {
    ss << fmt::format("- required-coopmat    : ?\n");
  }

  if (req) {
    if (req->required_shared_memory() == 0) {
      ss << fmt::format("- required-sh-mem     : 0B\n");
    } else {
      ss << fmt::format("- required-sh-mem     : {:.3f}KB\n",
                        static_cast<float>(req->required_shared_memory()) /
                            1e3f);
    }
  } else {
    ss << fmt::format("- requires-sh-mem     : ?\n");
  }

  const denox::dnx::PushConstant *pc = dispatch->push_constant();
  if (pc) {
    ss << fmt::format("- pc-size             : {}B\n", pc->size());
  } else {
    ss << fmt::format("- pc-size             : ?\n");
  }

  {
    ss << fmt::format("- sets                : {}\n",
                      dispatch->bindings()->size());
  }

  uint32_t binding_count = 0;
  for (uint32_t i = 0; i < dispatch->bindings()->size(); ++i) {
    binding_count = dispatch->bindings()->Get(i)->bindings()->size();
  }

  {
    ss << fmt::format("- bindings            : {}\n", binding_count);
  }
}

static void dispatches(const denox::dnx::Model *dnx, std::stringstream &ss) {
  ss << fmt::format("Dispatch-Schedule:\n");
  for (uint32_t i = 0; i < dnx->dispatches()->size(); ++i) {
    dispatch(dnx, i, ss);
  }
}

static denox::memory::Dtype deserialize_type(denox::dnx::ScalarType type) {
  switch (type) {
  case denox::dnx::ScalarType_I16:
  case denox::dnx::ScalarType_U16:
    throw std::runtime_error("invalid dtype");
  case denox::dnx::ScalarType_I32:
    return denox::memory::Dtype::I32;
  case denox::dnx::ScalarType_U32:
    return denox::memory::Dtype::U32;
  case denox::dnx::ScalarType_I64:
    return denox::memory::Dtype::I64;
  case denox::dnx::ScalarType_U64:
    return denox::memory::Dtype::U64;
  case denox::dnx::ScalarType_F16:
    return denox::memory::Dtype::F16;
  case denox::dnx::ScalarType_F32:
    return denox::memory::Dtype::F32;
  case denox::dnx::ScalarType_F64:
    return denox::memory::Dtype::F64;
  }
}

static void compute_requirements(const denox::dnx::Model *dnx,
                                 std::stringstream &ss) {
  std::set<uint32_t> subgroupSizes;
  std::vector<denox::CoopmatShape> requiredShapes;
  uint32_t max_sh_size = 0;

  for (uint32_t i = 0; i < dnx->dispatches()->size(); ++i) {
    const auto *dispatch = dnx->dispatches()->Get(i);
    const auto *req = dispatch->requirements();
    if (!req)
      continue;
    if (req->fixed_subgroup_size() != 0) {
      subgroupSizes.insert(req->fixed_subgroup_size());
    }
    const auto *coopmats = req->required_coopmat_shapes();
    for (const auto *coopmat : *coopmats) {
      if (coopmat) {
        auto atype = deserialize_type(coopmat->atype());
        auto btype = deserialize_type(coopmat->btype());
        auto ctype = deserialize_type(coopmat->ctype());
        auto acctype = deserialize_type(coopmat->acctype());

        auto it = std::ranges::find_if(
            requiredShapes, [&](const denox::CoopmatShape &shape) -> bool {
              return coopmat->m() == shape.M && coopmat->k() == shape.K &&
                     coopmat->n() == shape.N && atype == shape.atype &&
                     btype == shape.btype && ctype == shape.ctype &&
                     acctype == shape.acctype &&
                     coopmat->saturating() == shape.saturatingAccumulation;
            });
        if (it == requiredShapes.end()) {
          requiredShapes.push_back(denox::CoopmatShape{
              .M = coopmat->m(),
              .N = coopmat->n(),
              .K = coopmat->k(),
              .atype = atype,
              .btype = btype,
              .ctype = ctype,
              .acctype = acctype,
              .saturatingAccumulation = coopmat->saturating(),
              .subgroupScope = true,
          });
        }
      }
    }
    if (req->required_shared_memory()) {
      max_sh_size = std::max(max_sh_size, req->required_shared_memory());
    }
  }
  ss << fmt::format("Compute-Requirements:\n");
  {
    ss << fmt::format("- sh-size             : {:.1f}KB\n",
                      static_cast<float>(max_sh_size) / 1e3f);
  }

  if (subgroupSizes.empty()) {
    ss << fmt::format("- sg-control          : none\n");
  } else {
    std::string sg_sizes = "[";
    bool first = true;
    for (const auto &sg : subgroupSizes) {
      if (!first) {
        sg_sizes += ", ";
      }
      sg_sizes += fmt::format("{}", sg);
      first = false;
    }
    sg_sizes += "]";

    ss << fmt::format("- sg-control          : {}\n", sg_sizes);
  }
  {
    std::string str = "[";
    for (uint32_t i = 0; i < requiredShapes.size(); ++i) { 
      if (i != 0) {
        str += ", ";
      }
      const auto& s = requiredShapes[i];
      std::string type;
      switch (s.atype.kind()) {
      case denox::memory::DtypeKind::F16:
        type = "f16";
        break;
      case denox::memory::DtypeKind::F32:
        type = "f32";
        break;
      case denox::memory::DtypeKind::F64:
        type = "f64";
        break;
      case denox::memory::DtypeKind::U32:
        type = "u32";
        break;
      case denox::memory::DtypeKind::I32:
        type = "i32";
        break;
      case denox::memory::DtypeKind::U64:
        type = "u64";
        break;
      case denox::memory::DtypeKind::I64:
        type = "i64";
        break;
      }
      str += fmt::format("{}x{}x{}{}", s.M, s.K, s.N, type);
    }
    str += "]";
    ss << fmt::format("- coopmats            : {}\n", str);

  }    
}

void dump_dnx(DumpAction &action) {

  const denox::dnx::Model *dnx = denox::dnx::GetModel(action.dnx.data.data());

  std::stringstream ss;
  model_info(dnx, ss);
  compilation_info(dnx, ss);
  dispatches(dnx, ss);
  compute_requirements(dnx, ss);
  binary_size(dnx, action.dnx.data.size(), ss);

  std::cerr << ss.str();
}
