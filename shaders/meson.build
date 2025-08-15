
vkcnn_shaders_src = files(
  'src/vkcnn/shaders/conv/IConvShader.cpp',
  'src/vkcnn/shaders/conv/DirectConvShader.cpp',
  'src/vkcnn/shaders/copy/CopyTransformShader.cpp',
  'src/vkcnn/shaders/copy/ICopyShader.cpp',
  'src/vkcnn/shaders/activation/ActivationShader.cpp',
  'src/vkcnn/shaders/activation/IActivationShader.cpp',
  'src/vkcnn/shaders/upsample/DirectUpsampleShader.cpp',
  'src/vkcnn/shaders/upsample/IUpsampleShader.cpp',
  'src/vkcnn/shaders/pool/IPoolShader.cpp',
  'src/vkcnn/shaders/pool/DirectPoolShader.cpp',
  'src/vkcnn/shaders/preprocessing.cpp',
)

vkcnn_shaders_inc = include_directories('src')

vkcnn_shaders_lib = static_library('vkcnn-shaders',
  vkcnn_shaders_src,
  dependencies: [vkcnn_common],
  include_directories: vkcnn_shaders_inc,
  install: false,
  pic : true
)

vkcnn_shaders = declare_dependency(
  link_with: vkcnn_shaders_lib,
  include_directories: vkcnn_shaders_inc,
  dependencies: [vkcnn_common],
)
