cmake_minimum_required(VERSION 3.24)
project(denox VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS true)

include(cmake/colorful.cmake)

if (NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "") 
  set(CMAKE_BUILD_TYPE "Release")
endif()


include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

include(cmake/vulkan.cmake)
include(cmake/glslang.cmake)
include(cmake/spirv-tools.cmake)
include(cmake/fmt.cmake)
include(cmake/spdlog.cmake)
include(cmake/onnx.cmake)
include(cmake/generator.cmake)

include(cmake/warnings.cmake)

add_library(denox_compiler 

  compiler/compiler/cano/cano.cpp
  compiler/compiler/dce.cpp
  compiler/compiler/impl/ComputeDispatchBuilder.cpp
  compiler/compiler/impl/impl.cpp
  compiler/compiler/spec.cpp
  compiler/compiler/lifeness.cpp

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
  compiler/shaders/conv/DirectConvShader.cpp
  compiler/shaders/copy/CopyTransformShader.cpp
  compiler/shaders/pad/MemoryPadShader.cpp
  compiler/shaders/pool/BasicPoolShader.cpp
  compiler/shaders/slice/MemorySliceShader.cpp
  compiler/shaders/upsample/BasicUpsampleShader.cpp
  compiler/shaders/GlslCompiler.cpp
  compiler/shaders/global_glslang_runtime.cpp
  compiler/shaders/preprocessing.cpp

  compiler/api.cpp
  compiler/entry.cpp
)


target_compile_features(denox_compiler PUBLIC cxx_std_20)
add_library(denox::compiler ALIAS denox_compiler)
add_library(denox::denox ALIAS denox_compiler)


target_compile_definitions(denox_compiler 
  PRIVATE 
    SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE
    SPDLOG_NO_ATOMIC_LEVELS
    SPDLOG_DISABLE_DEFAULT_LOGGER
  )

target_compile_features(denox_compiler PUBLIC cxx_std_20)
set_target_properties(denox_compiler PROPERTIES CXX_STANDARD 20)
set_target_properties(denox_compiler PROPERTIES CXX_STANDARD_REQUIRED ON)

target_include_directories(denox_compiler
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/compiler> 
    $<INSTALL_INTERFACE:include/compiler>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/compiler
)

target_link_libraries(denox_compiler
  PUBLIC 
    denox::fmt
    denox::spdlog
    denox::onnx
    denox::generator
    denox::vulkan
    denox::glslang
    denox::spirv-tools
)


add_executable(denox
  cli/main.cpp
)



set_target_properties(denox PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(denox 
    PUBLIC 
      denox::compiler
)

option(DENOX_ENABLE_STRICT_WARNINGS "Enables most warning flags" ON)
if (DENOX_ENABLE_STRICT_WARNINGS)
  denox_enable_strict_warnings(denox)
  denox_enable_strict_warnings(denox_compiler)
endif()

# ================= Testing ======================

option(BUILD_TESTING "Build tests" ON)
if (BUILD_TESTING)
  include(cmake/googletest.cmake)

  enable_testing()

  add_executable(denox_test EXCLUDE_FROM_ALL
    test/main.cpp
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
  )


  if (DENOX_ENABLE_STRICT_WARNINGS)
    denox_enable_strict_warnings(denox_test)
  endif()


  target_link_libraries(denox_test
    PUBLIC
      denox::compiler
      denox::gtest
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
  


# ========== PACKAGE-CONFIG-STUFF =============

# install(TARGETS denox_compiler
#   EXPORT denoxTargets
#   ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#   RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#   INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
# )
#
# install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#
# write_basic_package_version_file(
#   "${CMAKE_CURRENT_BINARY_DIR}/denoxConfigVersion.cmake"
#   VERSION ${PROJECT_VERSION}
#   COMPATIBILITY AnyNewerVersion
# )
#
# configure_package_config_file(
#   cmake/denoxConfig.cmake.in
#   "${CMAKE_CURRENT_BINARY_DIR}/denoxConfig.cmake"
#   INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/denox
# )
#
# install(FILES
#   "${CMAKE_CURRENT_BINARY_DIR}/denoxConfig.cmake"
#   "${CMAKE_CURRENT_BINARY_DIR}/denoxConfigVersion.cmake"
#   DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/denox
# )
#
# install(EXPORT denoxTargets
#   FILE denoxTargets.cmake
#   NAMESPACE denox::
#   DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/denox
# )
