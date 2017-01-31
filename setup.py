#!/usr/bin/env python

import numpy

import setuptools

from distutils import core
from Cython.Build import cythonize

core.setup(
  ext_modules=cythonize(
      ["AlphaGo/preprocessing/rollout_feature.pyx"],
      language="c++",
      extra_compile_args=["-std=c++11"],
      extra_link_args=["-std=c++11"]
  ),
  include_dirs=[numpy.get_include()]
)


requires = [
]


setuptools.setup(
  name='RocAlphaGo',
  version='0.0.1',
  author='take',
  url='',
  packages=[
    'AlphaGo', 'AlphaGo.models', 'AlphaGo.preprocessing', 'AlphaGo.training', 'interface'
  ],
  scripts=[
    'scripts/rag', 'scripts/ragc'
  ],
  install_requires=requires,
  license='MIT',
  test_suite='test',
  classifiers=[
    'Operating System :: OS Independent',
    'Environment :: Console',
    'Programming Language :: Python',
    'License :: OSI Approved :: MIT License',
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Information Technology',
    'Intended Audience :: Science/Research',
    'Topic :: Utilities',
  ],
  data_files=[
    ('/usr/local/share/rocalphago/', ['model/policy.json']),
    ('/usr/local/share/rocalphago/', ['model/policy.hdf5']),
    ('/usr/local/share/rocalphago/', ['model/value.json']),
    ('/usr/local/share/rocalphago/', ['model/value.hdf5']),
  ]
)
