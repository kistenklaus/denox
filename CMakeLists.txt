cmake_minimum_required(VERSION 3.24)
project(denox VERSION 0.1.0 LANGUAGES C CXX)

include(cmake/colorful.cmake)

option(BUILD_PYTHON_BINDING "Build tests" OFF)
option(BUILD_TESTING "Build tests" ON)

option(DENOX_ENABLE_STRICT_WARNINGS "Enables most warning flags" ON)

option(DENOX_EXTERNALLY_MANAGED_VULKAN_CONTEXT 
  "Exposes vulkan as a public dependency and allows passing a vulkan instance and device, which will be used for device queries" 
  ON)

option(DENOX_SAN "Enables sanitizers" OFF)

set(BUILD_PIL BUILD_PYTHON_BINDING)

if (NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "") 
  set(CMAKE_BUILD_TYPE "Release")
endif()


# set(FETCHCONTENT_QUIET FALSE) <- super unstable
# set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
# set(FETCHCONTENT_FULLY_DISCONNECTED ON) 

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

include(cmake/cli11.cmake)
include(cmake/vulkan.cmake)
include(cmake/spirv-tools.cmake)
include(cmake/glslang.cmake)
include(cmake/fmt.cmake)
include(cmake/spdlog.cmake)
include(cmake/onnx.cmake)
include(cmake/generator.cmake)
include(cmake/flatbuffers.cmake)
include(cmake/dnx.cmake)
include(cmake/vma.cmake)

include(cmake/warnings.cmake)

add_library(denox_compiler EXCLUDE_FROM_ALL

  compiler/algorithm/binary_op_permutations.cpp

  compiler/compiler/cano/cano.cpp
  compiler/compiler/placement/placement.cpp
  compiler/compiler/dce.cpp
  compiler/compiler/impl/ComputeDispatchBuilder.cpp
  compiler/compiler/impl/impl.cpp
  compiler/compiler/spec.cpp
  compiler/compiler/lifeness.cpp
  compiler/compiler/sym_compile.cpp
  compiler/compiler/sym_table.cpp

  compiler/device_info/query/query_driver_device_info.cpp
  compiler/device_info/query/select_physical_device.cpp
  compiler/device_info/query/query_resource_limits.cpp
  compiler/device_info/query/query_layout_rules.cpp
  compiler/device_info/query/query_memory_model_properties.cpp
  compiler/device_info/query/query_coopmat_properties.cpp
  compiler/device_info/query/query_subgroup_properties.cpp
  compiler/device_info/query/create_query_instance.cpp

  compiler/diag/summary.cpp
  compiler/diag/failed_to_realize.cpp

  compiler/dnx/serialize.cpp

  compiler/memory/dtype/dtype_reference.cpp
  compiler/memory/dtype/f16.cpp

  compiler/memory/tensor/ActivationTensor.cpp
  compiler/memory/tensor/BiasTensor.cpp
  compiler/memory/tensor/FilterTensor.cpp

  compiler/symbolic/SymGraphEval.cpp
  compiler/symbolic/SymGraph_add_entry.cpp
  compiler/symbolic/SymGraph_affine.cpp
  compiler/symbolic/SymGraph_affine_symbol_cache.cpp
  compiler/symbolic/SymGraph_api.cpp
  compiler/symbolic/SymGraph_compile.cpp
  compiler/symbolic/SymGraph_debug.cpp
  compiler/symbolic/SymGraph_div_entry.cpp
  compiler/symbolic/SymGraph_eval.cpp
  compiler/symbolic/SymGraph_helpers.cpp
  compiler/symbolic/SymGraph_max_entry.cpp
  compiler/symbolic/SymGraph_min_entry.cpp
  compiler/symbolic/SymGraph_mod_entry.cpp
  compiler/symbolic/SymGraph_modsolver.cpp
  compiler/symbolic/SymGraph_modsolver_cache.cpp
  compiler/symbolic/SymGraph_mul_entry.cpp
  compiler/symbolic/SymGraph_nonaffine.cpp
  compiler/symbolic/SymGraph_nonaffine_symbol_cache.cpp
  compiler/symbolic/SymGraph_sub_entry.cpp

  compiler/frontend/onnx/details/ops/abs.cpp
  compiler/frontend/onnx/details/ops/add.cpp
  compiler/frontend/onnx/details/ops/average_pool.cpp
  compiler/frontend/onnx/details/ops/cast.cpp
  compiler/frontend/onnx/details/ops/ceil.cpp
  compiler/frontend/onnx/details/ops/clip.cpp
  compiler/frontend/onnx/details/ops/concat.cpp
  compiler/frontend/onnx/details/ops/constant.cpp
  compiler/frontend/onnx/details/ops/constant_of_shape.cpp
  compiler/frontend/onnx/details/ops/conv.cpp
  compiler/frontend/onnx/details/ops/div.cpp
  compiler/frontend/onnx/details/ops/expand.cpp
  compiler/frontend/onnx/details/ops/floor.cpp
  compiler/frontend/onnx/details/ops/gather.cpp
  compiler/frontend/onnx/details/ops/leaky_relu.cpp
  compiler/frontend/onnx/details/ops/max.cpp
  compiler/frontend/onnx/details/ops/max_pool.cpp
  compiler/frontend/onnx/details/ops/min.cpp
  compiler/frontend/onnx/details/ops/mod.cpp
  compiler/frontend/onnx/details/ops/mul.cpp
  compiler/frontend/onnx/details/ops/neg.cpp
  compiler/frontend/onnx/details/ops/pad.cpp
  compiler/frontend/onnx/details/ops/pow.cpp
  compiler/frontend/onnx/details/ops/reciprocal.cpp
  compiler/frontend/onnx/details/ops/relu.cpp
  compiler/frontend/onnx/details/ops/reshape.cpp
  compiler/frontend/onnx/details/ops/resize.cpp
  compiler/frontend/onnx/details/ops/round.cpp
  compiler/frontend/onnx/details/ops/shape.cpp
  compiler/frontend/onnx/details/ops/sign.cpp
  compiler/frontend/onnx/details/ops/slice.cpp
  compiler/frontend/onnx/details/ops/sqrt.cpp
  compiler/frontend/onnx/details/ops/squeeze.cpp
  compiler/frontend/onnx/details/ops/sub.cpp
  compiler/frontend/onnx/details/ops/transpose.cpp
  compiler/frontend/onnx/details/ops/unsqueeze.cpp

  compiler/io/fs/File.cpp
  compiler/io/fs/Path.cpp


  compiler/frontend/onnx/details/values/Attribute.cpp
  compiler/frontend/onnx/details/values/DeviceTensor.cpp
  compiler/frontend/onnx/details/values/Dtype.cpp
  compiler/frontend/onnx/details/values/HostTensor.cpp
  compiler/frontend/onnx/details/values/HostTensorStorage.cpp
  compiler/frontend/onnx/details/values/Tensor.cpp
  compiler/frontend/onnx/details/values/TensorViewDesc.cpp
  compiler/frontend/onnx/details/values/TensorShape.cpp
  compiler/frontend/onnx/details/values/Value.cpp
  compiler/frontend/onnx/details/import_node.cpp
  compiler/frontend/onnx/details/import_value_info.cpp
  compiler/frontend/onnx/onnx.cpp

  compiler/model/Model.cpp

  compiler/shaders/activation/BasicActivationShader.cpp
  compiler/shaders/conv/DirectConvShaderCM.cpp
  compiler/shaders/copy/CopyTransformShader.cpp
  compiler/shaders/pad/MemoryPadShader.cpp
  compiler/shaders/pool/BasicPoolShader.cpp
  compiler/shaders/slice/MemorySliceShader.cpp
  compiler/shaders/upsample/BasicUpsampleShader.cpp

  compiler/shaders/compiler/GlslCompiler.cpp
  compiler/shaders/compiler/GlslCompilerInstance.cpp
  compiler/shaders/compiler/ShaderPreprocessor.cpp
  compiler/shaders/compiler/global_glslang_runtime.cpp
  compiler/shaders/shaders.cpp

  compiler/api.cpp
  compiler/entry.cpp
)

if (DENOX_SAN) 
  target_compile_options(denox_compiler PRIVATE -fsanitize=undefined -fsanitize=address)
  target_link_options(denox_compiler PRIVATE -fsanitize=undefined -fsanitize=address)
endif()


target_compile_features(denox_compiler PUBLIC cxx_std_20)
set_target_properties(denox_compiler PROPERTIES CXX_STANDARD 20)
set_target_properties(denox_compiler PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(denox_compiler PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(denox_compiler
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/compiler> 
    $<INSTALL_INTERFACE:include/compiler>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/common> 
    $<INSTALL_INTERFACE:include/common>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/compiler
)


target_link_libraries(denox_compiler
  PRIVATE 
    denox::fmt
    denox::spdlog
    denox::onnx
    denox::generator
    denox::vulkan
    denox::spirv-tools
    denox::dnx
    denox::glslang

)

target_compile_definitions(denox_compiler 
  PUBLIC
    DENOX_VERSION=0
  PRIVATE 
    SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE
    SPDLOG_NO_ATOMIC_LEVELS
    SPDLOG_DISABLE_DEFAULT_LOGGER
)

target_compile_features(denox_compiler PUBLIC cxx_std_20)
add_library(denox::compiler ALIAS denox_compiler)



add_library(denox_runtime EXCLUDE_FROM_ALL
  runtime/api_context.cpp
  runtime/api_eval.cpp
  runtime/api_bench.cpp
  runtime/api_instance.cpp
  runtime/api_model.cpp
  runtime/api_queries.cpp
  runtime/context.cpp
  runtime/dnx_parse_helpers.cpp
  runtime/vma_impl.cpp
)

target_include_directories(denox_runtime PUBLIC
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/runtime> 
    $<INSTALL_INTERFACE:include/runtime>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/common> 
    $<INSTALL_INTERFACE:include/common>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
)

target_compile_definitions(denox_runtime 
  PUBLIC
    DENOX_VERSION=0
)

target_link_libraries(denox_runtime
  PRIVATE
    denox::fmt
    denox::vulkan
    denox::dnx
    denox::vma
)

target_compile_features(denox_runtime PUBLIC cxx_std_20)

set_target_properties(denox_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(denox::runtime ALIAS denox_runtime)


## Super temporary!!

add_executable(sandbox EXCLUDE_FROM_ALL
  sandbox/main.cpp
)
target_link_libraries(sandbox PRIVATE denox::runtime denox::fmt)


if (DENOX_EXTERNALLY_MANAGED_VULKAN_CONTEXT)
  target_link_libraries(denox_compiler PUBLIC denox::vulkan)
  target_compile_definitions(denox_compiler PUBLIC DENOX_EXTERNALLY_MANAGED_VULKAN_CONTEXT)
endif()


if (NOT BUILD_PYTHON_BINDING)
  add_executable(denox
    cli/main.cpp
  )

  if (DENOX_SAN) 
    target_compile_options(denox PRIVATE -fsanitize=undefined -fsanitize=address)
    target_link_options(denox PRIVATE -fsanitize=undefined -fsanitize=address)
  endif()

  set_target_properties(denox PROPERTIES POSITION_INDEPENDENT_CODE BUILD_PIL)

  target_link_libraries(denox 
      PUBLIC 
        denox::compiler
      PRIVATE
        denox::glslang
        denox::cli11
        
  )
  install(TARGETS denox DESTINATION bin)

  if (DENOX_ENABLE_STRICT_WARNINGS)
    denox_enable_strict_warnings(denox)
    denox_enable_strict_warnings(denox_compiler)
  endif()
endif()

# ================= Testing ======================

if (BUILD_TESTING)
  include(cmake/googletest.cmake)

  enable_testing()

  add_executable(denox_test EXCLUDE_FROM_ALL
    test/main.cpp
    test/compiler/algorithm/binary_op_permutations.cpp
    test/compiler/algorithm/match_first.cpp
    test/compiler/algorithm/match_all.cpp
    test/compiler/algorithm/shortest_dag_hyperpath.cpp
    test/compiler/memory/monotone_pool_allocator.cpp
    test/compiler/memory/linked_graph.cpp
    test/compiler/symbolic/affine.cpp
    test/compiler/symbolic/common.cpp
    test/compiler/symbolic/eval.cpp
    test/compiler/symbolic/modsolve.cpp
    test/compiler/symbolic/nonaffine.cpp
    test/compiler/symbolic/unsolvable.cpp
    test/compiler/symbolic/compile.cpp
  )


  if (DENOX_ENABLE_STRICT_WARNINGS)
    denox_enable_strict_warnings(denox_test)
  endif()


  target_link_libraries(denox_test
    PUBLIC
      denox::compiler
      denox::gtest
      denox::fmt
      denox::spdlog
      denox::onnx
      denox::generator
      denox::vulkan
      denox::glslang
      denox::spirv-tools
  )

  target_include_directories(denox_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/compiler
      ${CMAKE_CURRENT_SOURCE_DIR}/test
  )

  gtest_discover_tests(denox_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endif()

if (BUILD_PYTHON_BINDING)
  include(cmake/pybind11.cmake)

  target_compile_definitions(denox_runtime PRIVATE DENOX_QUIET)

  pybind11_add_module(_denox MODULE 
    bindings/python/denox/Module.cpp
    bindings/python/denox/Module_compile.cpp
    bindings/python/denox/Tensor.cpp
    bindings/python/denox/mod.cpp
  )
  target_link_libraries(_denox 
    PRIVATE 
      denox::compiler
      denox::runtime
      denox::fmt
  )
  target_compile_definitions(_denox PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)
  target_include_directories(_denox PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/dlpack)

  set_target_properties(_denox PROPERTIES POSITION_INDEPENDENT_CODE ON)


  if(APPLE)
    set_target_properties(_denox PROPERTIES 
      BUILD_RPATH "@loader_path" 
      INSTALL_RPATH "@loader_path"
    )
  elseif(UNIX)
    set_target_properties(_denox PROPERTIES 
      BUILD_RPATH "$ORIGIN" 
      INSTALL_RPATH "$ORIGIN"
    )
  endif()

  install(TARGETS _denox LIBRARY DESTINATION ${Python3_SITEARCH}/denox)

  # After installation, generate stubs next to installed .so
  install(CODE "
    message(STATUS \"Generating Python stubs for installed denox...\")
    execute_process(
      COMMAND \"${Python3_EXECUTABLE}\" -m pybind11_stubgen denox._denox
        -o ${Python3_SITEARCH}
      WORKING_DIRECTORY ${Python3_SITEARCH}
      RESULT_VARIABLE result
    )
    if(NOT result EQUAL 0)
      message(FATAL_ERROR \"pybind11-stubgen failed with exit code: ${result}\")
    endif()
  ")

endif()

