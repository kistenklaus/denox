
gtest_dep = dependency('gtest', required: true)

vkcnn_dev_tests_src = files(
  'src/main.cpp',
  'src/env.cpp',

  'src/ops/conv.cpp',
  'src/symbolic/affine.cpp',
  'src/symbolic/nonaffine.cpp',
  'src/symbolic/unsolvable.cpp',
  'src/symbolic/common.cpp',
  'src/symbolic/modsolve.cpp',
)

vkcnn_dev_tests = executable('vkcnn-tests',
  vkcnn_dev_tests_src,
  dependencies: [gtest_dep, vkcnn_dev_utils, vkcnn_runtime, vkcnn_shaders],
  include_directories: include_directories('src')
)

test('vkcnn-tests', vkcnn_dev_tests)
