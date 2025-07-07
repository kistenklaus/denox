
vkcnn_common_src = files(
  'src/vkcnn/common/io/read_file.cpp',
  'src/vkcnn/common/tensor/FilterHostTensor.cpp',
  'src/vkcnn/common/tensor/ActivationHostTensor.cpp'
)

vkcnn_common_inc = include_directories('src')

vkcnn_common_lib = static_library('vkcnn-common',
  vkcnn_common_src,
  dependencies: [],
  include_directories: vkcnn_common_inc,
  install: false,
  pic : true
)

vkcnn_common = declare_dependency(
  link_with: vkcnn_common_lib,
  include_directories: vkcnn_common_inc,
)
