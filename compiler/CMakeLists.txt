
### Copy shaders into build directory
set(COMPILER_SHADER_FILES 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/activation/basic_activation.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/conv/direct_conv.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/conv/direct_conv_cm.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/copy/copy_transform.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/pad/memory_pad.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/pool/basic_pool.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/slice/memory_slice.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/upsample/basic_upsample.comp
)

add_custom_target(denox_compiler_shaders ALL)

foreach(shader IN LISTS COMPILER_SHADER_FILES)
    file(RELATIVE_PATH rel
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${shader}
    )

    add_custom_command(
        TARGET denox_compiler_shaders
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${shader}
                ${CMAKE_BINARY_DIR}/share/denox/compiler/${rel}
        VERBATIM
    )
endforeach()


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

    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/canonicalize/canonicalize.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/lifeness/lifeness.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/specialization/specialization.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/dce/dce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/dce/prune_dead_supergraph.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/implement.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/OpImpl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/ComputeDispatchBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/shaders.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/noop/NoOp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/activation/BasicActivationShader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/conv/DirectConvShaderCM.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/conv/DirectConvShader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/copy/CopyTransformShader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/pad/MemoryPadShader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/pool/BasicPoolShader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/slice/MemorySliceShader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/implement/shaders/upsample/BasicUpsampleShader.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/selection/selection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/selection/selection.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/placement/placement.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/compile_shaders/compile_shaders.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/rebind_descriptors/rebind_descriptors.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/compile_symbols/compile_symbols.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/serialize/serialize.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/assumed_symeval/assumed_symeval.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/compile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/compiler/populate.cpp
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
    denox_spirv
    denox_database
)

add_dependencies(denox_compiler denox_compiler_shaders)


target_compile_features(denox_compiler PUBLIC cxx_std_20)
set_target_properties(denox_compiler PROPERTIES CXX_STANDARD 20)
set_target_properties(denox_compiler PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(denox_compiler PROPERTIES POSITION_INDEPENDENT_CODE ON)
if (DENOX_UNITY_BUILD) 
  set_target_properties(denox_compiler PROPERTIES UNITY_BUILD ON)
endif()

if (NOT DENOX_DISABLE_WARNINGS)
  denox_enable_strict_warnings(denox_compiler)
endif()

if (DENOX_SAN) 
  target_compile_options(denox_compiler PRIVATE -fsanitize=undefined -fsanitize=address)
  target_link_options(denox_compiler PRIVATE -fsanitize=undefined -fsanitize=address)
endif()
