
merian_proj = subproject('merian')
merian = merian_proj.get_variable('merian_dep')

vkcnn_runtime_src = files(
  'src/vkcnn/runtime/conv/ConvPipeline.cpp',
  'src/vkcnn/runtime/tensor/ActivationDeviceTensor.cpp',
  'src/vkcnn/runtime/tensor/FilterDeviceTensor.cpp'
)

vkcnn_runtime_inc = include_directories('src')

vkcnn_runtime_lib = static_library('vkcnn-runtime',
  vkcnn_runtime_src,
  dependencies: [vkcnn_common, merian, vkcnn_shaders],
  include_directories: vkcnn_runtime_inc,
  install: false,
  pic : true
)

vkcnn_runtime = declare_dependency(
  link_with: vkcnn_runtime_lib,
  include_directories: vkcnn_runtime_inc,
  dependencies : [vkcnn_common, merian, vkcnn_shaders],
)
