

vkcnn_dev_utils_src = files(
  'src/vkcnn/dev/utils/merian.cpp'
)

vkcnn_dev_utils_inc = include_directories('src')

vkcnn_dev_utils_lib = static_library('vkcnn-dev-utils',
  vkcnn_dev_utils_src,
  dependencies: [vkcnn_common, vkcnn_runtime],
  include_directories: vkcnn_dev_utils_inc,
  install: false,
  pic : true
)

vkcnn_dev_utils = declare_dependency(
  link_with: vkcnn_dev_utils_lib,
  include_directories: vkcnn_dev_utils_inc,
  dependencies: [vkcnn_common, vkcnn_runtime]
)
