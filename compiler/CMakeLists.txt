include(${PROJECT_SOURCE_DIR}/cmake/onnx.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/dnx.cmake)

add_library(denox_compiler
  EXCLUDE_FROM_ALL
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/frontend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/model/Model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/onnx.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/import_node.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/import_value_info.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/values/Attribute.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/values/DeviceTensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/values/Dtype.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/values/HostTensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/values/HostTensorStorage.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/values/Tensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/values/TensorShape.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/values/TensorViewDesc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/values/Value.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/abs.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/add.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/average_pool.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/cast.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/ceil.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/clip.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/concat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/constant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/constant_of_shape.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/conv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/div.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/expand.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/floor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/gather.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/leaky_relu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/max.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/max_pool.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/min.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/mod.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/mul.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/neg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/pad.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/pow.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/reciprocal.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/relu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/reshape.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/resize.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/round.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/shape.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/sign.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/slice.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/sqrt.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/squeeze.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/sub.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/transpose.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/frontend/onnx/details/ops/unsqueeze.cpp


)

target_include_directories(denox_compiler
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(denox_compiler 
  PRIVATE
    denox::dnx
    denox::onnx
  PUBLIC 
    denox_common
    denox_symbolic
    denox_spirv
    denox_database
)


target_compile_features(denox_compiler PUBLIC cxx_std_20)
set_target_properties(denox_compiler PROPERTIES CXX_STANDARD 20)
set_target_properties(denox_common PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(denox_common PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (NOT DENOX_DISABLE_WARNINGS)
  denox_enable_strict_warnings(denox_compiler)
endif()

if (DENOX_SAN) 
  target_compile_options(denox_compiler PRIVATE -fsanitize=undefined -fsanitize=address)
  target_link_options(denox_compiler PRIVATE -fsanitize=undefined -fsanitize=address)
endif()
