project('torch-shim', 'cpp',
  default_options: [
    'cpp_std=c++17',
  ]
)

libtorch_root = meson.current_source_dir()

cuda_dep = dependency('cuda', required: false)
cuda_found = cuda_dep.found()

cuda_libdir = ''
cuda_includedir = ''
cuda_libs = []
cuda_compile_args = []

cuda_link_flags = []
if cuda_found
  cuda_path = ''
  cuda_home = run_command('bash', '-c', 'echo $CUDA_HOME', check: false).stdout().strip()
  if cuda_home != ''
    cuda_path = cuda_home
  else
    nvcc = run_command('which', 'nvcc', check: false)
    if nvcc.returncode() == 0
      cuda_path = nvcc.stdout().strip().split('/bin')[0]
    endif
  endif

  cuda_libdir = cuda_path / 'lib'
  cuda_includedir = cuda_path / 'include'
  cuda_libs = [
    libtorch_root / 'lib/libtorch_cuda.so',
    libtorch_root / 'lib/libc10_cuda.so',
    cuda_libdir / 'libnvrtc.so',
    cuda_libdir / 'libcudart.so',
  ]
  cuda_link_flags = [
      '-Wl,--no-as-needed', libtorch_root / 'lib/libtorch_cuda.so',
      '-Wl,-rpath,' + cuda_libdir,
  ]

  message('CUDA detected at ' + cuda_path)
else
  message('CUDA not found, building CPU-only version')
endif

libtorch_libs = [
  libtorch_root / 'lib/libtorch.so',
  libtorch_root / 'lib/libtorch_cpu.so',
  libtorch_root / 'lib/libc10.so',
  libtorch_root / 'lib/libkineto.a',
] + cuda_libs

torch_dep = declare_dependency(
  include_directories: [
    include_directories('include'),
    include_directories('include/torch/csrc/api/include'),
    include_directories(cuda_includedir)
  ],
  dependencies: cuda_found ? [cuda_dep] : [],
  link_args: [
    '-Wl,--no-as-needed', libtorch_root / 'lib/libtorch_cpu.so',
  ] + cuda_link_flags + [
    '-Wl,--as-needed',
    '-Wl,-rpath,' + libtorch_root / 'lib',
  ] + libtorch_libs,
  compile_args: [
    '-DUSE_C10D_GLOO',
    '-DUSE_C10D_NCCL',
    '-DUSE_DISTRIBUTED',
    '-DUSE_RPC',
    '-DUSE_TENSORPIPE',
  ] + cuda_compile_args
)

meson.override_dependency('libtorch', torch_dep)
