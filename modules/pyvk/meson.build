py = import('python').find_installation(pure: false)

pyvk_sources = files(
  '__init__.py',
  'core.py',
  'reflect.py',
  'target.py',
  'cpp_wrapper.py'
)

py.install_sources(pyvk_sources, subdir: 'pyvk')
