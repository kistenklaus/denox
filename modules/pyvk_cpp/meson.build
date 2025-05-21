
pyvk_cpp_src = files(
  'src/host/bindings/pyvk_cpp.cpp'
)

py.extension_module(
    '_pyvk_cpp',
    pyvk_cpp_src,
    dependencies: [merian, pybind11],
    install: true,
)
